import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

class TCAV:
    def __init__(self, model, layer_name, concept_images, random_images):
        self.model = model
        self.layer_name = layer_name
        self.concept_images = concept_images
        self.random_images = random_images
        self.cav = self.get_cav()

    def get_activations(self, images):
        intermediate_layer_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.layer_name).output)
        activations = intermediate_layer_model.predict(images)
        return activations.reshape((activations.shape[0], -1))  # 2D 배열로 변환

    def get_cav(self):
        concept_activations = self.get_activations(self.concept_images)
        random_activations = self.get_activations(self.random_images)
        X = np.concatenate([concept_activations, random_activations])
        y = np.concatenate([np.ones(len(concept_activations)), np.zeros(len(random_activations))])
        lr = LogisticRegression().fit(X, y)
        return lr.coef_.flatten()  # 1D 배열로 변환

    def get_tcav_score(self, images, labels):
        activations = self.get_activations(images)
        concept_sensitivities = np.dot(activations, self.cav.T)
        return np.mean(np.dot(labels, concept_sensitivities.T) > 0)

# 사용 예제
if __name__ == "__main__":
    # 모델 로드
    model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)

    # 개념 이미지와 랜덤 이미지 로드 (예제 데이터를 사용한 로드)
    concept_images = np.load('concept_images.npy')
    random_images = np.load('random_images.npy')

    # TCAV 인스턴스 생성
    tcav = TCAV(model, 'mixed10', concept_images, random_images)

    # 테스트 이미지와 레이블 로드 (예제 데이터를 사용한 로드)
    test_images = np.load('test_image.npy')
    test_labels = np.load('test_label.npy')

    # TCAV 점수 계산
    tcav_score = tcav.get_tcav_score(test_images, test_labels)
    print(f"TCAV Score: {tcav_score}")
