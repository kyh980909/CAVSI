import numpy as np
from skimage import filters
from scipy.ndimage.measurements import label
from skimage.measure import regionprops

def normalization(x):
  return (x-np.min(x))/(np.max(x)-np.min(x))

def energy_point_game(bbox, saliency_map): # GT 부분의 활성화 값 / 전체 활성화 값
  x1, y1, x2, y2 = bbox
  w, h = saliency_map.shape
  
  empty = np.zeros((w, h))
  empty[y1:y2, x1:x2] = 1
  mask_bbox = saliency_map * empty  
  
  energy_bbox =  mask_bbox.sum()
  energy_whole = saliency_map.sum()
  proportion = energy_bbox / energy_whole
  
  return proportion

def get_label(xml):
  p_size = xml.find('size')
  p_box = xml.find('object').find('bndbox')
  size = {'width':int(p_size.find('width').text),'height': int(p_size.find('height').text)}
  box = {'xmin':int(p_box.find('xmin').text), 'ymin' : int(p_box.find('ymin').text),'xmax': int(p_box.find('xmax').text),'ymax': int(p_box.find('ymax').text)}
  xmin, ymin, xmax, ymax = box['xmin'] / size['width'] * 224, box['ymin'] / size['height'] * 224, box['xmax'] / size['width'] * 224,box['ymax'] / size['height'] * 224
  w, h = xmax - xmin, ymax - ymin
  return {'xmin':xmin, 'ymin':ymin, 'xmax':xmax,'ymax':ymax,'w':w, 'h':h}

def otsu_binary(x):
    thr = filters.threshold_otsu(x)
    binary = x > thr
    return np.multiply(binary, 255)

def IoU(boxA, boxB):
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def bitwiseSimilarity(maskA, maskB, mode):
    if mode == 1:
      return np.sum(np.bitwise_and(maskA, maskB))/np.sum(maskA)
    elif mode == 2:
      return np.sum(np.bitwise_and(maskA, maskB))/np.sum(np.bitwise_or(maskA, maskB))
    else:
      raise print("Mode must be 1 or 2")
       

def generate_bbox(saliency_map, threshold):
    labeled, nr_objects = label(saliency_map > threshold)
    props = regionprops(labeled)
    
    init = props[0].bbox_area
    bbox = tuple(props[0].bbox)
    for b in props:
      if init < b.bbox_area:
          init = b.bbox_area
          bbox = tuple(b.bbox)

    return bbox