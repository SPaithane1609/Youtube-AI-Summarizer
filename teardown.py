import boto3

ENDPOINT_NAME = input("Enter endpoint name to delete: ").strip()

if ENDPOINT_NAME:
    client = boto3.client("sagemaker")
    try:
        client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        client.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
        print(f"Successfully deleted endpoint and config: {ENDPOINT_NAME}")
    except Exception as e:
        print(f"Error deleting endpoint: {e}")