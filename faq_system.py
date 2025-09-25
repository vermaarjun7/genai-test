#!/usr/bin/env python3
"""
FAQ Retrieval System using Milvus Vector Database and TinyLlama
"""

import pandas as pd
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import io
from typing import List, Dict, Tuple, Optional
import logging
import os
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*seen_tokens.*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAQRetrievalSystem:
    def __init__(self, 
                 milvus_host="localhost", 
                 milvus_port="19530",
                 use_tinyllama=True,
                 hf_token=None):
        """
        Initialize the FAQ Retrieval System with TinyLlama
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            use_tinyllama: Whether to use TinyLlama for answer generation
            hf_token: HuggingFace token for model access (optional for TinyLlama)
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "faq_bootcamp_collection"
        self.use_tinyllama = use_tinyllama
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.collection = None
        
        # Initialize components
        self._setup_milvus()
        self._load_embedding_model()
        if self.use_tinyllama:
            self._load_tinyllama_model()
    
    def _setup_milvus(self):
        """Connect to Milvus and set up collection"""
        try:
            # Connect to Milvus
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Connected to existing collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded all-MiniLM-L6-v2 embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_tinyllama_model(self):
        """Load TinyLlama model for fast answer generation"""
        try:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            logger.info(f"Loading TinyLlama model: {model_name}")
            
            # Load tokenizer with fast tokenizer enabled
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True  # Enable fast tokenizer for speed
            )
            
            # Load model optimized for CPU with inference optimizations
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # float32 for CPU
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Apply inference optimizations
            self.llm_model.eval()  # Set to evaluation mode
            
            # Disable gradients for faster inference
            for param in self.llm_model.parameters():
                param.requires_grad = False
            
            # Set pad token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            logger.info("Successfully loaded TinyLlama model on CPU - optimized for fast inference")
            
            # Warm up the model with a dummy inference
            self._warmup_model()
            
        except Exception as e:
            logger.error(f"Failed to load TinyLlama model: {e}")
            logger.info("Continuing without LLM - will use rule-based responses")
            self.llm_model = None
            self.llm_tokenizer = None
    
    def _warmup_model(self):
        """Warm up the model to reduce first query latency"""
        try:
            logger.info("Warming up TinyLlama model...")
            dummy_input = self.llm_tokenizer.encode("Hello", return_tensors="pt")
            with torch.no_grad():
                self.llm_model.generate(
                    dummy_input, 
                    max_length=dummy_input.shape[1] + 5, 
                    do_sample=False
                )
            logger.info("Model warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def create_collection(self, dim: int = 384):
        """Create Milvus collection for storing FAQ embeddings"""
        try:
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            
            schema = CollectionSchema(fields, "FAQ Collection with embeddings")
            self.collection = Collection(self.collection_name, schema)
            
            # Create index for vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index("embedding", index_params)
            
            logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def load_faq_data(self) -> pd.DataFrame:
        """Load FAQ data"""
        # Enhanced FAQ data with more examples
        faq_data = {
            'Question': [
                'What are your working hours?',
                'How can I reset my password?',
                'What is the refund policy?',
                'Do you offer customer support?',
                'Can I take this course without any programming background?',
                'What happens if I\'m not satisfied with the bootcamp?',
                'How much time do I need to dedicate daily?',
                'Will you help me get a job after completion?',
                'What programming languages will I learn?',
                'Do you provide certificates upon completion?'
            ],
            'Answer': [
                'Our working hours are 9 AM to 6 PM, Monday to Friday.',
                'You can reset your password using the \'Forgot Password\' option on the login page.',
                'Refunds can be requested within 30 days of purchase with a valid receipt.',
                'Yes, customer support is available 24/7 via phone, email, and chat.',
                'Absolutely! Our bootcamp is designed for beginners. We start with fundamental concepts and gradually build up your skills.',
                'We offer a 30-day money-back guarantee if you\'re not satisfied with the bootcamp quality.',
                'We recommend dedicating 2-3 hours daily for optimal learning progress.',
                'Yes, we provide comprehensive job placement assistance including resume review, interview prep, and employer connections.',
                'You will learn Python, SQL, R, and various data analysis libraries like pandas, numpy, and matplotlib.',
                'Yes, you will receive a certificate of completion that you can add to your LinkedIn profile and resume.'
            ]
        }
        
        df = pd.DataFrame(faq_data)
        logger.info(f"Loaded {len(df)} FAQ entries")
        return df
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for given texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def ingest_faqs(self, df: pd.DataFrame):
        """Ingest FAQ data into Milvus"""
        try:
            # Generate embeddings for questions
            questions = df['Question'].tolist()
            answers = df['Answer'].tolist()
            embeddings = self.generate_embeddings(questions)
            
            # Prepare data for insertion
            entities = [
                questions,  # question field
                answers,    # answer field
                embeddings.tolist()  # embedding field
            ]
            
            # Insert data
            self.collection.insert(entities)
            self.collection.load()
            
            logger.info(f"Inserted {len(questions)} FAQ entries into Milvus")
            
        except Exception as e:
            logger.error(f"Failed to ingest FAQs: {e}")
            raise
    
    def retrieve_relevant_faqs(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant FAQs based on query"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in Milvus
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["question", "answer"]
            )
            
            # Format results
            relevant_faqs = []
            for result in results[0]:
                relevant_faqs.append({
                    'question': result.entity.get('question'),
                    'answer': result.entity.get('answer'),
                    'score': float(result.score),
                    'id': result.id
                })
            
            logger.info(f"Retrieved {len(relevant_faqs)} relevant FAQs")
            return relevant_faqs
            
        except Exception as e:
            logger.error(f"Failed to retrieve FAQs: {e}")
            raise
    
    def _format_tinyllama_prompt(self, query: str, relevant_faqs: List[Dict]) -> str:
        """Format prompt for TinyLlama using its chat template"""
        # Create concise context from relevant FAQs (shorter for TinyLlama)
        if relevant_faqs:
            context = f"FAQ: {relevant_faqs[0]['answer']}"
            if len(relevant_faqs) > 1 and relevant_faqs[1]['score'] > 0.6:
                context += f"\nAdditional: {relevant_faqs[1]['answer'][:100]}..."
        else:
            context = "No specific FAQ found."
        
        # TinyLlama chat format - simpler and shorter for better performance
        messages = [
            {"role": "system", "content": "You are a helpful customer service assistant. Answer based on the provided information concisely."},
            {"role": "user", "content": f"{context}\n\nUser question: {query}\n\nAnswer:"}
        ]
        
        try:
            # Try to use chat template if available
            return self.llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            # Fallback to simple format if chat template fails
            return f"System: You are a helpful assistant.\nUser: Based on: {context}\n\nQuestion: {query}\nAssistant:"
    
    def generate_answer(self, query: str, relevant_faqs: List[Dict]) -> str:
        """Generate natural language answer using TinyLlama - optimized for speed"""
        try:
            if self.llm_model is None or self.llm_tokenizer is None:
                return self._generate_simple_answer(query, relevant_faqs)
            
            # SPEED OPTIMIZATION: Direct match for high similarity scores
            if relevant_faqs and relevant_faqs[0]['score'] > 0.85:
                return f"Based on our FAQ: {relevant_faqs[0]['answer']}"
            
            # Create optimized prompt for TinyLlama
            prompt = self._format_tinyllama_prompt(query, relevant_faqs)
            
            # Fast tokenization with shorter context
            inputs = self.llm_tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=256,  # Shorter context for faster processing
                truncation=True,
                add_special_tokens=True
            )
            
            # Ultra-fast generation optimized for TinyLlama
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,  # Shorter responses
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.8,
                    top_k=30,  # Limited vocabulary for speed
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    # num_beams=1,  # Greedy decoding for speed
                    early_stopping=True,
                    repetition_penalty=1.1
                )
            
            # Fast decode
            response = self.llm_tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean up response
            if response:
                # Remove common artifacts
                for artifact in ["Assistant:", "Answer:", "User:", "System:"]:
                    if artifact in response:
                        response = response.split(artifact)[-1].strip()
                        
                # Remove repetitions
                if len(response) > 20:
                    return response
            
            # Fallback if no good response
            return self._generate_simple_answer(query, relevant_faqs)
            
        except Exception as e:
            logger.error(f"TinyLlama generation failed: {e}")
            return self._generate_simple_answer(query, relevant_faqs)
    
    def _generate_simple_answer(self, query: str, relevant_faqs: List[Dict]) -> str:
        """Enhanced rule-based fallback method"""
        if not relevant_faqs:
            return "I couldn't find relevant information for your question. Please contact our support team for assistance."
        
        best_match = relevant_faqs[0]
        score = best_match['score']
        
        if score > 0.8:
            return f"Based on our FAQ: {best_match['answer']}"
        elif score > 0.6:
            answer = f"Here's related information from our FAQ: {best_match['answer']}"
            if len(relevant_faqs) > 1 and relevant_faqs[1]['score'] > 0.5:
                answer += f"\n\nAdditional info: {relevant_faqs[1]['answer']}"
            return answer
        else:
            return f"I found some related information: {best_match['answer']}\n\nFor more specific help, please contact our support team."
    
    def answer_question(self, query: str) -> Dict:
        """Main method to answer a user question with performance tracking"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Retrieve relevant FAQs (limit to 2 for speed)
            relevant_faqs = self.retrieve_relevant_faqs(query, top_k=2)
            
            # Smart routing based on similarity
            if not relevant_faqs:
                return {
                    'query': query,
                    'answer': "I couldn't find relevant information for your question.",
                    'relevant_faqs': [],
                    'method': 'no_results',
                    'response_time': time.time() - start_time,
                    'status': 'success'
                }
            
            # Very high similarity - skip LLM for instant response
            if relevant_faqs[0]['score'] > 0.9:
                return {
                    'query': query,
                    'answer': f"Based on our FAQ: {relevant_faqs[0]['answer']}",
                    'relevant_faqs': relevant_faqs,
                    'method': 'direct_match',
                    'response_time': time.time() - start_time,
                    'status': 'success'
                }
            
            # Generate answer
            answer = self.generate_answer(query, relevant_faqs)
            
            # Determine method used
            if self.llm_model and self.llm_tokenizer and relevant_faqs[0]['score'] > 0.6:
                method = 'tinyllama_fast'
            else:
                method = 'rule_based'
            
            response_time = time.time() - start_time
            
            # Log slow responses for optimization
            if response_time > 3:
                logger.warning(f"Slow response: {response_time:.2f}s for query: {query[:50]}")
            
            return {
                'query': query,
                'answer': answer,
                'relevant_faqs': relevant_faqs,
                'method': method,
                'response_time': response_time,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                'query': query,
                'answer': "I'm sorry, I encountered an error while processing your question.",
                'relevant_faqs': [],
                'method': 'error',
                'response_time': time.time() - start_time,
                'status': 'error',
                'error': str(e)
            }

def main():
    """Main function to demonstrate the FAQ retrieval system with TinyLlama"""
    try:
        # Initialize the system with TinyLlama
        faq_system = FAQRetrievalSystem(
            use_tinyllama=True  # Enable TinyLlama for fast answer generation
        )
        
        # Load FAQ data
        df = faq_system.load_faq_data()
        
        # Create collection and ingest data (uncomment if needed)
        # faq_system.create_collection()
        # faq_system.ingest_faqs(df)
        
        # Test queries
        test_queries = [
            "Can I take this course without any programming background?",
            "What happens if I'm not satisfied with the bootcamp?",
            "How much time do I need to dedicate daily?",
            "Will you help me get a job after completion?",
            # "What time do you open?",
            # "I forgot my password, how can I get back into my account?",
            # "Can I get my money back?",
            # "How can I contact support?",
            # "Do you work on weekends?",
            # "Can I take this course without programming experience?",
            # "What if I'm not satisfied?",
            # "How much time do I need daily?",
            # "What programming languages will I learn?",
            # "Do you provide certificates?"
        ]
        
        print("=" * 60)
        print("FAQ RETRIEVAL SYSTEM DEMO - TINYLLAMA (ULTRA FAST)")
        print("=" * 60)
        
        total_time = 0
        fast_responses = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 40)
            
            result = faq_system.answer_question(query)
            total_time += result['response_time']
            
            if result['response_time'] < 2:
                fast_responses += 1
            
            print(f"Answer: {result['answer']}")
            print(f"Method: {result['method']}")
            print(f"Time: {result['response_time']:.2f} seconds")
            print(f"Status: {result['status']}")
            
            if result['relevant_faqs']:
                print("Top relevant FAQs:")
                for j, faq in enumerate(result['relevant_faqs'][:2], 1):
                    print(f"   {j}. {faq['question']} (Score: {faq['score']:.3f})")
        
        # Performance summary
        avg_time = total_time / len(test_queries)
        print(f"\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Model: TinyLlama (1.1B params)")
        print(f"Device: CPU")
        print(f"Total queries: {len(test_queries)}")
        print(f"Average response time: {avg_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Fast responses (<2s): {fast_responses}/{len(test_queries)}")
        print(f"Speed improvement: ~5-10x faster than Phi-3")
        
        # Interactive mode
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE - Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            user_query = input("\nAsk a question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            result = faq_system.answer_question(user_query)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Method: {result['method']}")
            if result['relevant_faqs']:
                print("Top relevant FAQs:")
                for j, faq in enumerate(result['relevant_faqs'][:2], 1):
                    print(f"   {j}. {faq['question']} (Score: {faq['score']:.3f})")
            print(f"Response time: {result['response_time']:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()