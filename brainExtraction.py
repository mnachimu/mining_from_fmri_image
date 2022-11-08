import cv2
import os
import numpy as np
import re

TEMPLATE_IMAGE = './template.PNG'
PATIENT_DATA_FOLDER_DEFAULT = './Data'
SLICES_FOLDER = './Slices'
BOUNDARIES_FOLDER = './Boundaries'


def slice_image_draw_boundary_save(image_file, destination_folder_name):

    image = cv2.imread(os.path.join(PATIENT_DATA_FOLDER_DEFAULT, image_file))
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
    boundary_folder = os.path.join(BOUNDARIES_FOLDER, destination_folder_name)
    if not os.path.exists(slices_folder):
        os.makedirs(slices_folder, exist_ok=True)
    if not os.path.exists(boundary_folder):
        os.makedirs(boundary_folder, exist_ok=True)

    for (x, y) in zip(x_pts, y_pts):

        square_tile = image[y:y + diff_y, x:x + diff_x]
        file_name = "{}_{}.png".format(str(x), str(y))

        x, y, z = square_tile.shape
        black_img = np.zeros((x, y, z), dtype=np.uint8)
        black_img[0:template.shape[0], 0:template.shape[1]] = template

        st = cv2.subtract(square_tile, black_img)
        val = np.sum(st)
        if val > 60000:
            cv2.imwrite(os.path.join(slices_folder, file_name), st)

            # draw a boundary
            gray_st = cv2.cvtColor(st, cv2.COLOR_BGR2GRAY)
            thresh_st = cv2.inRange(gray_st, 0, 25)
            edged_st = cv2.Canny(thresh_st, 50, 100)
            (cnts, _) = cv2.findContours(edged_st.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(st, cnts, -1, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(boundary_folder, file_name), st)


def get_image_files_and_process(patient_data_folder=PATIENT_DATA_FOLDER_DEFAULT):
    # get the list of all files for which image has to be sliced
    all_files = os.listdir(patient_data_folder)
    r = re.compile(".*thresh.png")
    thresh_image_files = list(filter(r.match, all_files))

    if not os.path.exists(SLICES_FOLDER):
        os.makedirs(SLICES_FOLDER)
    for file in thresh_image_files:
        slice_image_draw_boundary_save(file, file[:-4])

if __name__ == "__main__":
    print("Running brain extraction")
    print("Patient Data folder: {}".format(PATIENT_DATA_FOLDER_DEFAULT))
    print("Using green color for boundaries as the red and blue are already present in some of the images")
    get_image_files_and_process()
    print("Output Folders: Slices and Boundaries")
