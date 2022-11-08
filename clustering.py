import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cv2
import os
import numpy as np
import pandas as pd
import re
from sklearn.cluster import DBSCAN

TEMPLATE_IMAGE = './template.PNG'
PATIENT_DATA_FOLDER_DEFAULT = './testPatient'
SLICES_FOLDER = './Slices'
CLUSTERS_FOLDER = './Clusters'
EPSILON = 4
MIN_SAMPLES = 20


def slice_image_draw_boundary_save(image_file, destination_folder_name, patient_data_folder=PATIENT_DATA_FOLDER_DEFAULT):
    image = cv2.imread(os.path.join(patient_data_folder, image_file))
    template = cv2.imread(TEMPLATE_IMAGE)

    # using cv2.matchTemplate for creating tiles
    simpler_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    simpler_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    match = cv2.matchTemplate(simpler_image, simpler_template, cv2.TM_CCOEFF_NORMED)
    (y_pts, x_pts) = np.where(match >= 0.9)

    x_pts_list = list(set(x_pts))
    x_pts_list.sort()
    diff_x = x_pts_list[1] - x_pts_list[0]
    y_pts_list = list(set(y_pts))
    y_pts_list.sort()
    diff_y = y_pts_list[1] - y_pts_list[0]

    slices_folder = os.path.join(SLICES_FOLDER, destination_folder_name)
    clusters_folder = os.path.join(CLUSTERS_FOLDER, destination_folder_name)
    if not os.path.exists(slices_folder):
        os.makedirs(slices_folder, exist_ok=True)
    if not os.path.exists(clusters_folder):
        os.makedirs(clusters_folder, exist_ok=True)
    df = pd.DataFrame(columns=['SliceNumber', 'ClusterCount'])
    slice_number = 1
    for (x, y) in zip(x_pts, y_pts):
        square_tile = image[y:y + diff_y, x:x + diff_x]
        file_name = "{}_{}_{}.png".format(str(slice_number), str(x), str(y))

        x_s, y_s, z_s = square_tile.shape
        black_img = np.zeros((x_s, y_s, z_s), dtype=np.uint8)
        black_img[0:template.shape[0], 0:template.shape[1]] = template

        st = cv2.subtract(square_tile, black_img)
        val = np.sum(st)
        if val > 60000:
            cv2.imwrite(os.path.join(slices_folder, file_name), st)

            # draw a boundary
            # gray_st = cv2.cvtColor(st, cv2.COLOR_BGR2GRAY)
            # thresh_st = cv2.inRange(gray_st, 0, 25)
            # edged_st = cv2.Canny(thresh_st, 50, 100)
            # (cnts, _) = cv2.findContours(edged_st.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(st, cnts, -1, (0, 255, 0), 1)
            # cv2.imwrite(os.path.join(clusters_folder, file_name), st)

            # find clusters
            st_frame = cv2.GaussianBlur(st, (3, 3), 0)
            st_hsv = cv2.cvtColor(st_frame, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([100, 128, 100])
            upper_blue = np.array([215, 255, 255])
            # Threshold the HSV image to get only blue colors
            mask_blue = cv2.inRange(st_hsv, lower_blue, upper_blue)

            lower_red1 = np.array([0, 128, 100])
            upper_red1 = np.array([60, 255, 255])
            # Threshold the HSV image to get only red colors
            mask_red_1 = cv2.inRange(st_hsv, lower_red1, upper_red1)

            lower_red2 = np.array([165, 128, 100])
            upper_red2 = np.array([179, 255, 255])
            # Threshold the HSV image to get only red colors
            mask_red_2 = cv2.inRange(st_hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

            # cv2.imshow('masks ', np.concatenate(
            #     (np.concatenate((mask_red_1, mask_red_2), axis=1), np.concatenate((mask_blue, mask_red), axis=1))
            #     , axis=0))

            mask = cv2.bitwise_or(mask_red, mask_blue)
            res = st.copy()
            res = cv2.bitwise_and(res, res, mask=mask)
            res_grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            np_pixel_ind = np.transpose(np.nonzero(res_grey))

            dbscan_cluster = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES)
            if len(np_pixel_ind) == 0:
                df = df.append({'SliceNumber': slice_number, 'ClusterCount': 0},
                               ignore_index=True)
                slice_number = slice_number + 1
                continue
            clus = dbscan_cluster.fit_predict(np_pixel_ind)
            clus_freq = {}
            for cluster in clus:
                clus_freq[cluster] = clus_freq.get(cluster, 0) + 1
            qualified_clusters = set()
            for cluster in clus:
                if clus_freq[cluster] >= 135:
                    qualified_clusters.add(cluster)
            c = 0
            # print(qualified_clusters)
            res_copy = res.copy()
            for i, j in np_pixel_ind:
                if clus[c] in qualified_clusters:
                    res_copy[i][j] = [0, 255, 255]
                if not clus[c] in qualified_clusters or clus[c] == -1:
                    res_copy[i][j] = [0, 200, 200]
                # elif clus[c] in qualified_clusters:
                #     res_copy[i][j] = [0, 255, 255]
                # else:
                #     res_copy[i][j] = [255, 255, 255]
                c = c + 1

            # cv2.imshow('Result - {ORIGINAL : RED/BLUE PICK : CLUSTERED} (Note: only yellow are valid clusters)',
            #            np.concatenate((st, res, res_copy), axis=1))
            # cv2.waitKey()
            cv2.imwrite(os.path.join(clusters_folder, file_name), res_copy)
            df = df.append({'SliceNumber': slice_number, 'ClusterCount': len(qualified_clusters)}, ignore_index=True)
            slice_number = slice_number + 1
    print("Total count {}".format(df['ClusterCount'].sum()))
    # df.concat()
    df.to_csv(os.path.join(clusters_folder, "report.csv"))


def get_image_files_and_process(patient_data_folder=PATIENT_DATA_FOLDER_DEFAULT):
    # get the list of all files for which image has to be sliced
    all_files = os.listdir(patient_data_folder)
    r = re.compile(".*thresh.png")
    thresh_image_files = list(filter(r.match, all_files))

    if not os.path.exists(SLICES_FOLDER):
        os.makedirs(SLICES_FOLDER)
    for file in thresh_image_files:
        slice_image_draw_boundary_save(file, file[:-4], patient_data_folder)


if __name__ == "__main__":
    print("Running clustering")
    print("Patient Data folder: {}".format(PATIENT_DATA_FOLDER_DEFAULT))
    print("Using yellow color to display all clusters, including cluster's pixel count < 135")
    get_image_files_and_process()
    print("Output Folders: Slices and Clusters")
