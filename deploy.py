import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Initializing SageMaker session and role
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Hugging Face Hub configuration
hub_config = {
    'HF_MODEL_ID': 'facebook/bart-large-cnn',
    'HF_TASK': 'summarization'
}

# Defining the container image and model configuration
huggingface_model = HuggingFaceModel(
    env=hub_config,
    role=role,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310"
)

print("Deploying endpoint to Amazon SageMaker (approx. 3-5 minutes)...")
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"  # Free-tier eligible compute instance
)

print("=" * 60)
print(f"Endpoint Deployed Successfully: {predictor.endpoint_name}")
print("=" * 60)