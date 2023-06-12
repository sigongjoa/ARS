# ARS
던전엔 파이터 아바타 추천 시스템 AI 시스템 학습 및 배포 

## 데이터 전처리 및 학습

* 학습 데이터 준비
1. vit-gpt2-image-captioning.ipynb 노트북을 실행하여 metajson 파일을 생성합니다.
이 노트북은 데이터 전처리를 처리하고 metajson 파일을 준비합니다.
Hugging Face의 Accelerate를 설정하고 Lora 모델을 학습하는 방법은 다음 웹사이트를 참조하세요: 
> https://ngwaifoong92.medium.com/how-to-fine-tune-stable-diffusion-using-lora-85690292c6a8

* 추천 설정
추천 설정을 구성하려면 다음 단계를 수행하세요:

**stable_diffusion.ipynb** 노트북을 실행하여 이미지 사용 수준을 확인합니다.
이 노트북을 사용하여 이미지 사용의 임계값을 평가할 수 있습니다.

* listener
listener.ipynb 파일에서 다음 변경사항을 수행하세요:

다음 줄을 수정하세요: 
images = pipe(prompt=prompt, image=resized_image, guidance_scale=1, strength=0.25).images[0].
prompt, guidance_scale, strength 매개변수를 필요에 맞게 조정하세요.

* post
포즈를 가져오려면 다음 단계를 수행하세요:
flask_test.py 파일에서 포트를 설정하고 Flask를 실행하여 파일을 받으세요.

포트 설정을 구성하고 파일을 수신하기 위해 파일을 실행하세요.
`listener.ipynb` 파일에서 post_url 변수를 JSON을 전송할 경로로 업데이트하세요.

JSON 데이터를 전송할 경로를 지정하세요.
listener.ipynb 노트북을 실행하세요.
