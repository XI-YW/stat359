from instructor.final_project.arithmetic_llm.evaluator import ModelEvaluator, eval_expression

# Load model
evaluator = ModelEvaluator(
    model_path="models/instruction_lora_rank8/merged_model.pt",
    tokenizer_path="data/tokenizer"
)

# Your test expression
expr = "15"   # change this to anything you want

# Ground truth
gt = eval_expression(expr)['answer']

# Model output
prompt = f"Evaluate: {expr} <think>"
generated = evaluator._generate_solution(prompt)

# Extract result
pred = evaluator.extract_final_result(generated)

# Print nicely
print("\n===== SINGLE SAMPLE TEST =====")
print(f"Expression: {expr}")
print(f"Ground Truth: {gt}")
print(f"Predicted: {pred}")
print(f"Correct: {pred == gt}")
print("Generated Text:")
print(generated)
print("=============================\n")