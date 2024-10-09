import cv2  # OpenCV 라이브러리 가져오기: 이미지 처리, 결과 시각화, 직접 그리기와 텍스트 추가
from google.cloud import vision # Google Cloud의 Vision API를 사용하기 위한 라이브러리, 이미지 객체 감지하기 위해

# Google Vision API 클라이언트를 설정하는 함수.
def setup_vision_client():
    client = vision.ImageAnnotatorClient() #Google Cloud Vision API의 클라이언트를 생성
    return client #생성된 클라이언트를 반환하여, 다른 함수나 코드에서 이 클라이언트를 통해 Vision API 기능을 사용

# 객체 라벨의 이름에 따라 고유한 색상을 반환하는 함수.
def get_color_for_label(label):
    color_map = {
        'person': (255, 0, 0),  # 빨강
        'backpack': (0, 255, 0),  # 초록
        'clothing': (0, 0, 255),  # 파랑
        'pants': (255, 255, 0)  # 청록
    }
    return color_map.get(label.lower(), (255, 255, 255))  # 기본 색상: 흰색

# Vision API 클라이언트와 이미지 경로를 입력받아 객체를 분석하고 시각화
def analyze_image_with_vision_api(client, image_path):

    # 이미지 파일을 읽고 Vision API 형식으로 변환
    with open(image_path, 'rb') as image_file: #with 문을 사용하면 파일을 열고 자동으로 닫는 것을 보장, 분석할 이미지 파일은 텍스트가 아닌 바이너리 데이터로 저장됨, 그래서 바이너리 모드로 열기
        content = image_file.read() #이미지의 모든 바이너리 데이터를 읽고 저장
    image = vision.Image(content=content) #Google Cloud Vision API의 이미지 객체를 생성하는 코드

    # 객체 감지 요청
    response = client.object_localization(image=image) #Vision API의 object_localization 메소드를 호출하여 이미지에서 객체를 감지
    objects = response.localized_object_annotations #감지된 객체들의 정보를 objects 리스트에 저장

    # OpenCV를 사용하여 이미지 읽기
    img = cv2.imread(image_path)
    height, width, _ = img.shape #OpenCV를 사용하여 이미지를 읽고, 이미지의 높이와 너비 가져오기

    # 감지된 객체를 순회하며 경계 상자 그리기
    for object_ in objects:
        vertices = [
            (int(vertex.x * width), int(vertex.y * height))
            for vertex in object_.bounding_poly.normalized_vertices
        ] #감지된 각 객체의 각 정규화된 좌표(0과 1 사이의 값)를 실제 이미지의 픽셀 좌표로 변환

        if len(vertices) == 4:  # 사각형인 경우
            color = get_color_for_label(object_.name) #라벨에 따른 색상 선택:
            cv2.rectangle(img, vertices[0], vertices[2], color, 2)  # OpenCV에서 제공하는 함수로, 이미지에 사각형
            label = f"{object_.name}: {object_.score:.2f}" #label 변수는 객체의 이름과 신뢰 점수(0에서 1 사이의 값)
            cv2.putText(img, label, (vertices[0][0], vertices[0][1] - 10), #cv2.putText는 이미지 위에 텍스트 작성, 
                        #(vertices[0][0], vertices[0][1] - 10)는 텍스트 시작 위치를 지정하며, 경계 상자 위에 텍스트를 표시하기 위해 y좌표를 약간 위로 이동
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) #글꼴 스타일, 0.5는 글꼴 크기, color는 텍스트 색상, 2는 텍스트 두께

    # 결과 이미지를 파일로 저장
    result_image_path = '/Users/sin-yeonghyeon/Desktop/result_with_boxes.jpg'
    cv2.imwrite(result_image_path, img)

    # HTML 파일 내용 생성
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Analysis Result</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            img {{ max-width: 100%; height: auto; border: 2px solid #333; }}
        </style>
    </head>
    <body>
        <h1>Image Analysis Result</h1>
        <img src="{result_image_path}" alt="Analyzed Image">
        <h2>Detected Objects:</h2>
        <ul>
    """

    # 객체가 감지되지 않은 경우와 감지된 경우
    if not objects:
        html_content += "<li>No objects detected.</li>"
    else:
        for object_ in objects:
            html_content += f"<li>{object_.name} with confidence {object_.score:.2f}</li>"

    html_content += """
        </ul>
    </body>
    </html>
    """

    # HTML 파일을 저장, 생성된 HTML 콘텐츠를 파일로 저장하여, 웹 브라우저에서 결과를 봄
    with open("/Users/sin-yeonghyeon/Desktop/image_analysis_result.html", "w") as html_file:
        html_file.write(html_content)

# 이미지 파일 경로를 지정
image_path = '/Users/sin-yeonghyeon/Desktop/test10.jpg' #분석할 이미지의 파일 경로
vision_client = setup_vision_client()  # Vision API 클라이언트 설정
analyze_image_with_vision_api(vision_client, image_path)  # 이미지 분석 함수 호출, Vision API 클라이언트와 이미지 경로를 입력
