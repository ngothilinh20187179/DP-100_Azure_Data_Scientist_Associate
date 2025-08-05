from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

# --- 1: Connect to Azure ML Workspace ---
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

try:
    ml_client = MLClient.from_config(credential=credential)
except Exception as ex:
    subscription_id = "your azure subscription_id",
    resource_group = "your azure resource_group_name",
    workspace_name = "your azure workspace_name",
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# --- 2: Load from YAML & Register Component to Azure ML Workspace ---
add_component_loaded = load_component(source="./my_add_component/add_component.yaml")
ml_client.components.create_or_update(add_component_loaded)

# --- 3: Create Pipeline use component ---
# Get component
add_numbers_component = ml_client.components.get(name="add_numbers_component", version="1.0.0")
# định nghĩa một Azure Machine Learning Pipeline trong Python SDK v2. 
@pipeline(
    name="simple_add_pipeline",
    display_name="Simple Addition Pipeline",
    description="A pipeline to add two numbers using a custom component."
)
def add_pipeline(input_num1: int, input_num2: int):
    """
    Pipeline that takes two numbers and adds them using the 'add_numbers_component'.
    """
    add_step = add_numbers_component(
        num1=input_num1,
        num2=input_num2
    )
    return {
        "final_sum_output": add_step.outputs.sum_output
    }

pipeline_job = add_pipeline(input_num1=5, input_num2=7)
pipeline_job.experiment_name = "simple-component-pipeline-experiment"
pipeline_job.display_name = "Add 5 and 7 Example"
pipeline_job.settings.default_compute = "ngothilinhau129152" 

# --- Bước 4: Submit snd run Pipeline ---
returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"Monitor your pipeline job at: {returned_job.studio_url}")

# --- (Tùy chọn) Run Command Job ---
# from azure.ai.ml import command
# job_train = command(
#     code="./src", # Giả sử train.py nằm trong thư mục src ở thư mục gốc
#     command="python train.py --training_data ${{inputs.diabetes_data}} --reg_rate ${{inputs.reg_rate}}",
#     inputs={
#         "diabetes_data": Input(
#             type=AssetTypes.URI_FILE, 
#             path="azureml:training-data-uri-file:1" # Thay bằng ID/path dữ liệu của bạn
#         ),
#         "reg_rate": 0.01,
#     },
#     environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
#     compute="ngothilinhau129152", # Thay bằng tên compute của bạn
#     display_name="diabetes-train-job",
#     experiment_name="diabetes-train-experiment", 
#     tags={"model_type": "LogisticRegression"}
# )
# returned_train_job = ml_client.create_or_update(job_train)
# print("Monitor your training job at", returned_train_job.studio_url)