from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

# List of questions and answers in Thai
questions_and_answers = [
    {"question": "นักศึกษาสามารถลงทะเบียนเรียนได้กี่หน่วยกิต", "expected_response": "12-19"},
    {"question": "สามารถลงทะเบียนเรียนได้มากกว่า 22 หน่วยกิตได้หรือไม่", "expected_response": "ไม่ได้ นักศึกษาสามารถลงทะเบียนสูงสุดได้ 22 หน่วยกิต ทั้งนี้ต้องได้รับการอนุมัติจากอาจารย์ที่ปรึกษา"},
    {"question": "หากต้องการลงทะเบียนเรียนมากกว่า 19 หน่วยกิต ต้องทำอย่างไร", "expected_response": "ขออนุมัติจากอาจารย์ที่ปรึกษา"},
    {"question": "ภาคการศึกษาพิเศษ สามารถลงทะเบียนเรียนได้กี่หน่วยกิต", "expected_response": "ไม่เกิน 9 หน่วยกิต"},
    {"question": "หากมี GPAX ต่ำกว่า 2.00 สามารถสำเร็จการศึกษาได้หรือไม่", "expected_response": "ไม่สำเร็จ"},
    {"question": "ต้องมีชั่วโมงกิจกรรมเท่าไหร่จึงจะเรียนจบ", "expected_response": "ต้องมีชั่วโมงกิจกรรมมากกว่าหรือเท่ากับ 100 ตลอดหลักสูตรการศึกษา"},
    {"question": "เกรดเฉลี่ยสะสม 3.90 จะได้เกียรตินิยมอันดับที่เท่าไหร่", "expected_response": "เกียรตินิยมอันดับ 1"},
    {"question": "มีผลการเรียนเฉลี่ยสะสมมากกว่า 3.25 แต่เคยมีประวัติการได้เกรดตกอยู่ในเกณฑ์ Fe สามารถขอรับเกียรตินิยมได้หรือไม่", "expected_response": "ไม่สามารถขอรับเกียรตินิยมได้"},
    {"question": "นักศึกษาที่ลาพักการศึกษาแล้ว ต้องรายงานภายในกี่วันก่อนการลงทะเบียนเรียน", "expected_response": "ไม่น้อยกว่า 14 วันหรือ 2 สัปดาห์"},
    {"question": "หากเข้าเรียนต่ำกว่าร้อยละ 80 ของเวลาเรียนทั้งหมด จะสามารถมีสิทธิสอบได้หรือไม่", "expected_response": "ไม่มีสิทธิสอบ"}
]
def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="supachai/llama-3-typhoon-v1.5")
    # Ensure the model understands that answers should be in Thai
    prompt += "\nPlease provide the answer in Thai."

    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    # Print raw model response for debugging
    print(f"Model's response: {evaluation_results_str_cleaned}")

    # Check for true or false in the model's response
    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        # If the response is not "true" or "false", handle the case
        print("\033[93m" + f"Unexpected response format: {evaluation_results_str_cleaned}" + "\033[0m")
        return False  # Or raise an exception depending on your preference


def test_all_rules():
    correct_count = 0
    total_count = len(questions_and_answers)

    for idx, qa in enumerate(questions_and_answers):
        print(f"Testing question {idx + 1}/{total_count}: {qa['question']}")
        
        result = query_and_validate(
            question=qa["question"],
            expected_response=qa["expected_response"]
        )
        
        # Track the count of correct responses
        if result:
            correct_count += 1
        
        # Print a separator line after each question's result
        print("-" * 50)

    # Calculate and print overall accuracy
    accuracy = (correct_count / total_count) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

# def test_all_rules():
#     correct_count = 0
#     total_count = len(questions_and_answers)

#     for qa in questions_and_answers:
#         if query_and_validate(
#             question=qa["question"],
#             expected_response=qa["expected_response"]
#         ):
#             correct_count += 1

#     accuracy = (correct_count / total_count) * 100
#     print(f"\nAccuracy: {accuracy:.2f}%")

# Call the test function to run all tests
test_all_rules()
