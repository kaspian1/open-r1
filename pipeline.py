from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

# Load a smaller dataset for testing
dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(5))

# Use a smaller model to fit within GPU memory
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

with Pipeline(
    name="distill-qwen-1.5b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:
    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 2048,  # Reduced sequence length
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 1024,  # Reduced output length
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=2,  # Reduced number of generations
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )

if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="kas1/numina-deepseek-r1-qwen-1.5b")