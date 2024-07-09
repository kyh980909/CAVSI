import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2
from utils import *
from PIL import Image
from skimage.metrics import structural_similarity as ssim

class SISE():
    def __init__(self, model, model_name, img_path, class_idx, img_size=(224,224), grouping_thr=0.5, detail=0) -> None:
        self.model = model
        self.input_size = model.input_shape[1:3]
        self.model_name = model_name
        self.feature_maps = {}
        self.avg_grads = {}
        self.img = Image.open(img_path).resize(img_size)
        self.img_size= img_size
        img_arr = np.asarray(self.img)[:, :, :3] / 255.
        self.input_img = np.expand_dims(img_arr, 0)
        self.class_idx = class_idx
        self.filtered_feature_maps = {}
        self.postprocessed_feature_maps = {}
        self.layer_visualization_maps = {}
        self.total_reduction_rate = 0
        self.result = None
        self.group_bbox = {}
        self.grouping_thr = grouping_thr
        self.detail=detail
    
    def feature_extractor(self):
        if self.model_name=='vgg16':
            # Feature map을 추출할 layer 결정
            block = [1, 4, 8, 12, 16]
        elif self.model_name=='resnet50':
            # Feature map을 추출할 layer 결정
            block = [4, 38, 80, 142, 174]
        elif self.model_name=='resnet152':
            block = [4, 38, 120, 482, 514]
        elif self.model_name == 'inceptionv3':
            block = [9, 16, 86, 228, 310]
        else:
            assert print('Not support')

        outputs = [self.model.layers[i].output for i in block]
        feature_map_extraction_model = Model([self.model.inputs], outputs)

        # Layer별 feature map 추출
        feature_maps = {}
        feature_maps_list = feature_map_extraction_model.predict(self.input_img)

        for i, fmap in enumerate(feature_maps_list):
            feature_maps[f'conv{i}'] = tf.convert_to_tensor(np.squeeze(fmap))

        self.feature_maps = feature_maps

        

    def feature_filtering(self):
        # conv layer별 피쳐맵과 confidence score(softmax 값)의 gradient 계산
        if self.model_name == 'vgg16':
            block = [1, 4, 8, 12, 16, 22]
        elif self.model_name == 'resnet50':
            block = [2, 38, 80, 142, 174, 176]
        elif self.model_name=='resnet152':
            block = [4, 38, 120, 482, 514, 516]
        elif self.model_name=='inceptionv3':
            block = [9, 16, 86, 228, 310, 312]
        else:
            assert print('Not support')
        
        outputs = [self.model.layers[i].output for i in block]

        grad_model = Model([self.model.inputs], outputs)

        with tf.GradientTape(persistent=True) as tape:
            *conv_outputs, pred = grad_model(self.input_img)
            class_channel = pred[:, self.class_idx]

        grads = {}
        for i, conv in enumerate(conv_outputs):
            grads[f'conv{i}'] = tape.gradient(class_channel, conv)[0]

        # 피쳐맵의 평균 gradient 계산
        avg_grads = {}
        for k, v in grads.items():
            avg_grads[k] = tf.reduce_mean(v, axis=(0,1))

        self.avg_grads = avg_grads

        # 피쳐맵의 평균 gradient가 0이 넘는 피쳐맵만 필터링
        filtered_feature_maps = {}
        for k, v in avg_grads.items():
            transpose = tf.transpose(self.feature_maps[k], perm=[2,0,1])[v>0] # 필터링 용이를 위해 transpose
            filtered_feature_maps[k] = tf.transpose(transpose, perm=[1,2,0]) # transpose 다시 되돌리기

        # 필터링 된 피쳐맵 수 비교
        sum1 = sum2 = 0
        for k1, k2 in zip(avg_grads.values(), filtered_feature_maps.values()):
            if self.detail == 1:
                print(f'{len(k1)} -> {k2.shape[-1]}, {len(k1)-k2.shape[-1]}개 감소 (감소율: {(k2.shape[-1]-len(k1))/len(k1)*100}%)')
            sum1 += len(k1)
            sum2 += k2.shape[-1]

        if self.detail == 1:
            print('\nTotal')
            print(f'{sum1} -> {sum2}, {sum1-sum2}개 감소 (감소율: {(sum2-sum1)/sum1*100}%)')

        self.filtered_feature_maps = filtered_feature_maps
        self.total_reduction_rate = (sum2-sum1)/sum1*100
            
    def postprocess(self):
        # Bilinear interpolation
        postprocessed_feature_maps = {}
        for k in self.filtered_feature_maps.keys():
            for i in range(0, self.filtered_feature_maps[k].shape[2], 512):
                # i+512가 array.shape[2]보다 크거나 같으면, 슬라이싱할 수 없습니다.
                # 이 경우, array[:, :, i:]를 슬라이싱합니다.
                if i+512 >= self.filtered_feature_maps[k].shape[2]:
                    if k in postprocessed_feature_maps:
                        try:
                            postprocessed_feature_maps[k] = np.concatenate((postprocessed_feature_maps[k], cv2.resize(self.filtered_feature_maps[k][:, :, i:].numpy(), self.img_size, interpolation=cv2.INTER_LINEAR)), axis=2)
                        except Exception:
                            postprocessed_feature_maps[k] = np.concatenate((postprocessed_feature_maps[k], np.expand_dims(cv2.resize(self.filtered_feature_maps[k][:, :, i:].numpy(), self.img_size, interpolation=cv2.INTER_LINEAR), 2)), axis=2)
                    else:
                        postprocessed_feature_maps[k] = cv2.resize(self.filtered_feature_maps[k][:, :, i:].numpy(), self.img_size, interpolation=cv2.INTER_LINEAR)
                else:
                    if k in postprocessed_feature_maps:
                        try:
                            postprocessed_feature_maps[k] = np.concatenate((postprocessed_feature_maps[k], cv2.resize(self.filtered_feature_maps[k][:, :, i:i+512].numpy(), self.img_size, interpolation=cv2.INTER_LINEAR)), axis=2)
                        except Exception:
                            postprocessed_feature_maps[k] = np.concatenate((postprocessed_feature_maps[k], np.expand_dims(cv2.resize(self.filtered_feature_maps[k][:, :, i:i+512].numpy(), self.img_size, interpolation=cv2.INTER_LINEAR), 2)), axis=2)
                    else:
                        postprocessed_feature_maps[k] = cv2.resize(self.filtered_feature_maps[k][:, :, i:i+512].numpy(), self.img_size, interpolation=cv2.INTER_LINEAR)

            for i in range(postprocessed_feature_maps[k].shape[-1]):
                if np.max(postprocessed_feature_maps[k][:,:,i]) == 0.0:
                    continue
                postprocessed_feature_maps[k][:,:,i] = (postprocessed_feature_maps[k][:,:,i] - np.min(postprocessed_feature_maps[k][:,:,i]))/(np.max(postprocessed_feature_maps[k][:,:,i])-np.min(postprocessed_feature_maps[k][:,:,i]))

        self.postprocessed_feature_maps = postprocessed_feature_maps
    
    def filtering_zero_feature_maps(self):
        # Calculate sum of all feature maps
        sum_featurempas = {}
        for k, v in self.filtered_feature_maps.items():
            sum_featurempas[k] = tf.reduce_sum(v, axis=(0,1))

        # Sum of all feature maps != 0 filtering
        not_zero_feature_maps = {}
        for k, v in sum_featurempas.items():
            transpose = tf.transpose(self.filtered_feature_maps[k], perm=[2,0,1])[v!=0] # 필터링 용이를 위해 transpose
            not_zero_feature_maps[k] = tf.transpose(transpose, perm=[1,2,0]) # transpose 다시 되돌리기

        # 필터링 된 피쳐맵 수 비교
        sum1 = sum2 = 0
        for k1, k2 in zip(self.avg_grads.values(), not_zero_feature_maps.values()):
            if self.detail == 1:
                print(f'{len(k1)} -> {k2.shape[-1]}, {len(k1)-k2.shape[-1]}개 감소 (감소율: {(k2.shape[-1]-len(k1))/len(k1)*100}%)')
            sum1 += len(k1)
            sum2 += k2.shape[-1]

        if self.detail == 1:
            print('\nTotal')
            print(f'{sum1} -> {sum2}, {sum1-sum2}개 감소 (감소율: {(sum2-sum1)/sum1*100}%)\n')

        self.filtered_feature_maps = not_zero_feature_maps

    def attribution_masks_compress(self):
        layers = ['conv3', 'conv4']

        layer_bbox = {}

        # conv3, conv4 레이어의 feature map들 bbox 좌표 계산
        for layer in layers:
            layer_bbox[layer] = []
            for index in range(self.postprocessed_feature_maps[layer].shape[2]):
                binary = otsu_binary(self.postprocessed_feature_maps[layer][:,:,index])
                labeled, nr_objects = label(binary > 0)
                props = regionprops(labeled)

                init = props[0].bbox_area
                bbox = tuple(props[0].bbox)
                for b in props:
                    if init < b.bbox_area:
                        init = b.bbox_area
                        bbox = tuple(b.bbox)

                layer_bbox[layer].append(bbox)

        # IoU가 0.5 이상인 feature map끼리 grouping
        group_bbox = {}
        for k in layer_bbox.keys():
            temp = layer_bbox[k].copy()
            group_bbox[k] = []
            for i in range(len(temp)):
                if temp[i] == 0:
                    continue
                temp_group = [i]
                for j in range(i+1, len(temp)):
                    if temp[j] == 0:
                        continue
                    if IoU(temp[i], temp[j]) >= self.grouping_thr:
                        temp_group.append(j)
                        temp[j] = 0
                temp[i] = 0
                group_bbox[k].append(temp_group)

        self.group_bbox = group_bbox

        compressed_feature_maps = {}

        for layer in layers:
            for b in group_bbox[layer]:
                compressed_feature_map = np.zeros_like(self.postprocessed_feature_maps[layer][:,:,0].shape)
                for i, feature_map_index in enumerate(b):
                    if i == 0:
                        compressed_feature_map = self.postprocessed_feature_maps[layer][:,:, feature_map_index]
                    else:
                        compressed_feature_map += self.postprocessed_feature_maps[layer][:,:, feature_map_index]

                if layer in compressed_feature_maps:
                    compressed_feature_maps[layer] = np.concatenate((compressed_feature_maps[layer], np.expand_dims(normalization(compressed_feature_map), axis=2)), axis=2)
                else:
                    compressed_feature_maps[layer] = np.expand_dims(normalization(compressed_feature_map), axis=2)

            self.postprocessed_feature_maps[layer] = compressed_feature_maps[layer]

        # 필터링 된 피쳐맵 수 비교
        sum1 = sum2 = 0
        for k1, k2 in zip(self.avg_grads.values(), self.postprocessed_feature_maps.values()):
            if self.detail == 1:
                print(f'{len(k1)} -> {k2.shape[-1]}, {len(k1)-k2.shape[-1]}개 감소 (감소율: {(k2.shape[-1]-len(k1))/len(k1)*100}%)')
            sum1 += len(k1)
            sum2 += k2.shape[-1]
        if self.detail == 1:
            print('\nTotal')
            print(f'{sum1} -> {sum2}, {sum1-sum2}개 감소 (감소율: {(sum2-sum1)/sum1*100}%)')
        self.total_reduction_rate = (sum2-sum1)/sum1*100

    def new_attribution_masks_compress1(self, mode):
        layers = ['conv3', 'conv4']

        layer_bbox = {}

        # conv3, conv4 레이어의 feature map들 bbox 좌표 계산
        for layer in layers:
            layer_bbox[layer] = []
            for index in range(self.postprocessed_feature_maps[layer].shape[2]):
                binary = otsu_binary(self.postprocessed_feature_maps[layer][:,:,index])
                layer_bbox[layer].append(binary)

        # IoU가 0.5 이상인 feature map끼리 grouping
        group_bbox = {}
        for k in layer_bbox.keys():
            temp = layer_bbox[k].copy()
            group_bbox[k] = []
            for i in range(len(temp)):
                if isinstance(temp[i], np.ndarray) == False:
                    continue
                temp_group = [i]
                for j in range(i+1, len(temp)):
                    if isinstance(temp[j], np.ndarray) == False:
                        continue
                    if bitwiseSimilarity(temp[i], temp[j], mode) >= self.grouping_thr:
                        temp_group.append(j)
                        temp[j] = 0
                temp[i] = 0
                group_bbox[k].append(temp_group)

        self.group_bbox = group_bbox

        compressed_feature_maps = {}

        for layer in layers:
            for b in group_bbox[layer]:
                compressed_feature_map = np.zeros_like(self.postprocessed_feature_maps[layer][:,:,0].shape)
                for i, feature_map_index in enumerate(b):
                    if i == 0:
                        compressed_feature_map = self.postprocessed_feature_maps[layer][:,:, feature_map_index]
                    else:
                        compressed_feature_map += self.postprocessed_feature_maps[layer][:,:, feature_map_index]

                if layer in compressed_feature_maps:
                    compressed_feature_maps[layer] = np.concatenate((compressed_feature_maps[layer], np.expand_dims(normalization(compressed_feature_map), axis=2)), axis=2)
                else:
                    compressed_feature_maps[layer] = np.expand_dims(normalization(compressed_feature_map), axis=2)

            self.postprocessed_feature_maps[layer] = compressed_feature_maps[layer]

        # 필터링 된 피쳐맵 수 비교
        sum1 = sum2 = 0
        for k1, k2 in zip(self.avg_grads.values(), self.postprocessed_feature_maps.values()):
            if self.detail == 1:
                print(f'{len(k1)} -> {k2.shape[-1]}, {len(k1)-k2.shape[-1]}개 감소 (감소율: {(k2.shape[-1]-len(k1))/len(k1)*100}%)')
            sum1 += len(k1)
            sum2 += k2.shape[-1]

        if self.detail == 1:
            print('\nTotal')
            print(f'{sum1} -> {sum2}, {sum1-sum2}개 감소 (감소율: {(sum2-sum1)/sum1*100}%)')
        self.total_reduction_rate = (sum2-sum1)/sum1*100

    def new_attribution_masks_compress2(self):
        layers = ['conv3', 'conv4']

        # IoU가 0.5 이상인 feature map끼리 grouping
        group_fmaps = {}
        for k in layers:
            temp = self.postprocessed_feature_maps[k].copy()
            group_fmaps[k] = []
            while temp.shape[2] != 0:
                base_fmap = temp[:,:,0]
                temp = np.delete(temp, 0, axis=2)
                temp_group = [base_fmap]
                temp_delete_group = []

                for i in range(temp.shape[2]):
                    if ssim(base_fmap, temp[:,:,i], full=True)[0] >= self.grouping_thr:
                        temp_group.append(temp[:,:,i])
                        temp_delete_group.append(i)

                group_fmaps[k].append(temp_group)
                temp = np.delete(temp, temp_delete_group, axis=2)

        self.group_fmaps = group_fmaps

        compressed_feature_maps = {}

        for layer in layers:
            for fmaps in group_fmaps[layer]:
                compressed_feature_map = None
                for i, fmap in enumerate(fmaps):
                    if i == 0:
                        compressed_feature_map = fmap
                    else:
                        compressed_feature_map += fmap

                if layer in compressed_feature_maps:
                    compressed_feature_maps[layer] = np.concatenate((compressed_feature_maps[layer], np.expand_dims(normalization(compressed_feature_map), axis=2)), axis=2)
                else:
                    compressed_feature_maps[layer] = np.expand_dims(normalization(compressed_feature_map), axis=2)

            self.postprocessed_feature_maps[layer] = compressed_feature_maps[layer]

        # 필터링 된 피쳐맵 수 비교
        sum1 = sum2 = 0
        for k1, k2 in zip(self.avg_grads.values(), self.postprocessed_feature_maps.values()):
            if self.detail == 1:
                print(f'{len(k1)} -> {k2.shape[-1]}, {len(k1)-k2.shape[-1]}개 감소 (감소율: {(k2.shape[-1]-len(k1))/len(k1)*100}%)')
            sum1 += len(k1)
            sum2 += k2.shape[-1]
            
        if self.detail == 1:
            print('\nTotal')
            print(f'{sum1} -> {sum2}, {sum1-sum2}개 감소 (감소율: {(sum2-sum1)/sum1*100}%)')
        self.total_reduction_rate = (sum2-sum1)/sum1*100

    def generate_layer_visualization_map(self):
        layer_visualization_maps = {}
        for k in self.postprocessed_feature_maps.keys():
            masks = np.expand_dims(tf.transpose(self.postprocessed_feature_maps[k], perm=[2,0,1]), axis=-1)
            masked = self.input_img*masks
            preds = self.model.predict(masked)
            layer_visualization_maps[k] = preds.T.dot(masks.reshape(masks.shape[0],-1)).reshape(-1, self.img_size[0], self.img_size[1])

        self.layer_visualization_maps = layer_visualization_maps

    def layers_fusion(self):
        result = normalization(self.layer_visualization_maps['conv0'][self.class_idx]).copy()

        for k in self.layer_visualization_maps.keys():
            if k == 'conv0':
                continue
            result += self.layer_visualization_maps[k][self.class_idx]
            thr = filters.threshold_otsu(normalization(self.layer_visualization_maps[k][self.class_idx]))
            binary = normalization(self.layer_visualization_maps[k][self.class_idx]) > thr
            binary = np.multiply(binary, 255)
            result = result * binary

        self.result = result