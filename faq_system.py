#!/usr/bin/env python3
"""
FAQ Retrieval System using Milvus Vector Database and Llama 3
"""

import pandas as pd
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import io
from typing import List, Dict, Tuple, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAQRetrievalSystem:
    def __init__(self, 
                 milvus_host="localhost", 
                 milvus_port="19530",
                 llama_model_size="7B",
                 hf_token=None,
                 use_quantization=True):
        """
        Initialize the FAQ Retrieval System
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            llama_model_size: Llama model size ("7B", "11B", or "70B")
            hf_token: HuggingFace token for model access
            use_quantization: Whether to use 4-bit quantization to save memory
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "faq_collection"
        self.llama_model_size = llama_model_size
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.use_quantization = use_quantization
        
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.collection = None
        
        # Initialize components
        self._setup_milvus()
        self._load_embedding_model()
        self._load_llama3_model()
    
    def _setup_milvus(self):
        """Connect to Milvus and set up collection"""
        try:
            # Connect to Milvus
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
            
            # Check if collection exists and drop it for fresh start
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            
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
    
    def _load_llama3_model(self):
        """Load Llama 3 model for answer generation"""
        try:
            # Map model sizes to HuggingFace model names
            model_mapping = {
                "7B": "meta-llama/Meta-Llama-3-8B-Instruct",
                "8B": "meta-llama/Meta-Llama-3-8B-Instruct", 
                "11B": "meta-llama/Meta-Llama-3.1-11B-Instruct",
                "70B": "meta-llama/Meta-Llama-3.1-70B-Instruct"
            }
            
            model_name = model_mapping.get(self.llama_model_size)
            if not model_name:
                raise ValueError(f"Unsupported model size: {self.llama_model_size}")
            
            logger.info(f"Loading Llama 3 {self.llama_model_size} model: {model_name}")
            
            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Set up quantization config for memory efficiency
            quantization_config = None
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Using 4-bit quantization")
            
            # Load model
            model_kwargs = {
                "token": self.hf_token,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["low_cpu_mem_usage"] = True
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Set pad token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            logger.info(f"Successfully loaded Llama 3 {self.llama_model_size} model")
            
        except Exception as e:
            logger.error(f"Failed to load Llama 3 model: {e}")
            logger.info("Note: Make sure you have access to Llama models and provide a valid HuggingFace token")
            # Fallback to a simple response generation
            self.llm_model = None
            self.llm_tokenizer = None
    
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
        """Load FAQ data (in this case, we'll create it from the provided data)"""
        # Create the FAQ data as provided
        faq_data = {
            'Question': [
                'What are your working hours?',
                'How can I reset my password?',
                'What is the refund policy?',
                'Do you offer customer support?'
            ],
            'Answer': [
                'Our working hours are 9 AM to 6 PM, Monday to Friday.',
                'You can reset your password using the \'Forgot Password\' option on the login page.',
                'Refunds can be requested within 30 days of purchase with a valid receipt.',
                'Yes, customer support is available 24/7 via phone, email, and chat.'
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
    
    def _format_llama3_prompt(self, query: str, relevant_faqs: List[Dict]) -> str:
        """Format prompt for Llama 3 using proper chat template"""
        # Create context from relevant FAQs
        context = "FAQ Information:\n"
        for i, faq in enumerate(relevant_faqs, 1):
            context += f"{i}. Q: {faq['question']}\n   A: {faq['answer']}\n"
        
        # Create the system and user messages
        system_message = """You are a helpful customer service assistant. Use the provided FAQ information to answer the user's question accurately and helpfully. If the FAQ information doesn't directly answer the question, provide the most relevant information available and be honest about any limitations."""
        
        user_message = f"{context}\n\nUser Question: {query}\n\nPlease provide a helpful answer based on the FAQ information above."
        
        # Format using Llama 3 chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return self.llm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_answer(self, query: str, relevant_faqs: List[Dict]) -> str:
        """Generate natural language answer using Llama 3"""
        try:
            if self.llm_model is None or self.llm_tokenizer is None:
                # Fallback: Simple rule-based response
                return self._generate_simple_answer(query, relevant_faqs)
            
            # Format prompt for Llama 3
            prompt = self._format_llama3_prompt(query, relevant_faqs)
            
            # Tokenize
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=True
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response if response else self._generate_simple_answer(query, relevant_faqs)
            
        except Exception as e:
            logger.error(f"Failed to generate answer with Llama 3: {e}")
            return self._generate_simple_answer(query, relevant_faqs)
    
    def _generate_simple_answer(self, query: str, relevant_faqs: List[Dict]) -> str:
        """Fallback method for answer generation"""
        if not relevant_faqs:
            return "I'm sorry, I couldn't find relevant information to answer your question."
        
        # Return the most relevant FAQ answer with some context
        best_match = relevant_faqs[0]
        return f"Based on our FAQ, {best_match['answer']}"
    
    def answer_question(self, query: str) -> Dict:
        """Main method to answer a user question"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Retrieve relevant FAQs
            relevant_faqs = self.retrieve_relevant_faqs(query, top_k=3)
            
            # Generate answer
            answer = self.generate_answer(query, relevant_faqs)
            
            return {
                'query': query,
                'answer': answer,
                'relevant_faqs': relevant_faqs,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                'query': query,
                'answer': "I'm sorry, I encountered an error while processing your question.",
                'relevant_faqs': [],
                'status': 'error',
                'error': str(e)
            }

def main():
    """Main function to demonstrate the FAQ retrieval system"""
    try:
        # Initialize the system
        # You can change llama_model_size to "7B", "11B", or "70B"
        # Make sure to set HUGGINGFACE_TOKEN environment variable or pass hf_token parameter
        faq_system = FAQRetrievalSystem(
            llama_model_size="7B",  # Change this to "11B" or "70B" as needed
            use_quantization=True   # Set to False if you have enough GPU memory
        )
        
        # Load FAQ data
        df = faq_system.load_faq_data()
        
        # Create collection and ingest data
        faq_system.create_collection()
        faq_system.ingest_faqs(df)
        
        # Test queries
        test_queries = [
            "What time do you open?",
            "I forgot my password, how can I get back into my account?",
            "Can I get my money back?",
            "How can I contact support?",
            "Do you work on weekends?"
        ]
        
        print("=" * 60)
        print("FAQ RETRIEVAL SYSTEM DEMO - LLAMA 3")
        print("=" * 60)
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 40)
            
            result = faq_system.answer_question(query)
            
            print(f"📝 Answer: {result['answer']}")
            print(f"📊 Status: {result['status']}")
            
            if result['relevant_faqs']:
                print("🎯 Top relevant FAQs:")
                for i, faq in enumerate(result['relevant_faqs'][:2], 1):
                    print(f"   {i}. {faq['question']} (Score: {faq['score']:.3f})")
            
            print()
        
        # Interactive mode
        print("=" * 60)
        print("INTERACTIVE MODE - Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            user_query = input("\n💬 Ask a question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            result = faq_system.answer_question(user_query)
            print(f"\n🤖 {result['answer']}")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()