import numpy as np
import cv2
import glob

img_list = glob.glob('images/*.png')
img_list = img_list[:10]
imgs = []
for i in range(len(img_list)):
    imgs.append(cv2.imread(img_list[i]))
hists = []
for i, img in enumerate(imgs):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    hists.append(hist)

methods = {'INTERSECT': cv2.HISTCMP_INTERSECT, }
all_compare = []
for l in range(len(hists)):
    query = hists[l]
    part_compare = []
    for j, (name, flag) in enumerate(methods.items()):
        feat_compare = []
        for i, (hist, img) in enumerate(zip(hists, imgs)):
            ret = cv2.compareHist(query, hist, flag)
            if flag == cv2.HISTCMP_INTERSECT:
                ret = ret / np.sum(query)
            print(f"img{l + 1} vs img{i + 1} : {ret * 100:.2f}%")
            feat_compare.append(ret)
        part_compare.append(feat_compare)
        print()

    all_compare.append(part_compare)

print(all_compare[0][0])
print(np.array(all_compare).shape)
