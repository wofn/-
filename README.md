#이미지 객체 감지 및 시각화
이 프로젝트는 OpenCV와 Google Cloud Vision API를 사용하여 이미지에서 객체를 감지하고, 감지된 객체를 이미지 위에 시각적으로 표시한 후, 결과를 HTML 파일로 저장합니다.

##주요 기능

###라이브러리 가져오기:
OpenCV로 이미지 처리 및 시각화를 수행합니다.
Google Cloud Vision API를 사용하여 이미지에서 객체를 감지합니다.

####Vision API 클라이언트 설정:
setup_vision_client 함수를 통해 Vision API 클라이언트를 생성하여 다양한 기능을 사용할 수 있도록 합니다.

####객체 라벨 색상 설정:
get_color_for_label 함수는 감지된 객체의 라벨에 따라 고유한 색상을 지정하여, 경계 상자의 색상을 결정합니다.

####이미지 분석 및 시각화:
analyze_image_with_vision_api 함수는 이미지를 읽어 Vision API를 통해 객체를 감지하고, OpenCV를 사용하여 경계 상자를 그려 이미지에 시각적으로 표시합니다. 결과 이미지는 파일로 저장됩니다.

####HTML 결과 생성:
감지된 객체 정보를 포함한 HTML 파일을 생성하여, 웹 브라우저에서 결과를 확인할 수 있습니다.
이 프로젝트는 이미지에서 특정 객체를 감지하고, 그 객체를 강조하여 시각적으로 표현하는 데 유용합니다.
