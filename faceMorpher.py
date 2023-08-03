import dlib
import cv2
import glob
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

class FaceMorpher:
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self,image, name):    
        # Detect the faces in image an get the landmarks
        # Select only a few landmarks
        # Add aditional landmarks (corners and top, bottom, left and right)

        rects = self.detector(image, 0)
        
        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects): 
            h,w = image.shape[:2]

            shape = self.predictor(image, rect)
            landmarks_points = []
            for n in range(0, 68):
                x = shape.part(n).x
                y = shape.part(n).y
                landmarks_points.append((x, y))

            # add more landmarks (corners)
            landmarks_points.append((1,1)) #69
            landmarks_points.append((1,h-1)) #70
            landmarks_points.append((w-1,1)) #71
            landmarks_points.append((w-1,h-1)) #72

            # add more landmarks (top, bottom, left and right)
            landmarks_points.append(((w-1)//2,1)) #73
            landmarks_points.append((1,(h-1)//2)) #74
            landmarks_points.append(((w-1)//2,h-1)) #75
            landmarks_points.append(((w-1),(h-1)//2)) #76
        return landmarks_points

    def triangulation(self,first_landmarks):
        # Perform Delaunay triangulation on all the landmarks
        triangles = Delaunay(first_landmarks).simplices
        return triangles

    def affine_transform(self, src, src_tri, dst_tri,size):
        # do nothing if src_triangles == dest_triangles
        # if np.all(src_tri == dst_tri):
        #     return src
        [x1, y1] = src_tri[0]
        [x2, y2] = src_tri[1]
        [x3, y3] = src_tri[2]

        [x1_prime, y1_prime] = dst_tri[0]
        [x2_prime, y2_prime] = dst_tri[1]
        [x3_prime, y3_prime] = dst_tri[2]
        
        inverse_warped_image = np.zeros([size[1],size[0],3])

        A = np.array([[x1,x2,x3],[y1,y2,y3],[1,1,1]])
        X = np.array([[x1_prime,x2_prime,x3_prime],[y1_prime,y2_prime,y3_prime],[1,1,1]])

        T = np.matmul(A,np.linalg.inv(X))

        # Iterate over all the pixels in the transformed image
        for x_prime in range(size[0]):
            for y_prime in range(size[1]):
                # Compute the inverse transformation for each pixel
                input_pixels = np.array([x_prime,y_prime,1])
                k = np.matmul(T,input_pixels)
                x = int(k[0])
                y = int(k[1])
                
                # Check if the point is within the bounds of the original image
                if x >= 0 and x < src.shape[1] and y >= 0 and y < src.shape[0]:
                    # Set the value of the inverse warped image at this location
                    inverse_warped_image[y_prime, x_prime] = src[y, x]
        return inverse_warped_image

    def morph_triangle(self,im, im_out, src_tri, dst_tri):
        # morph each triangle in the image
        # Get bounding boxes around triangles
        src_rect = cv2.boundingRect(np.float32([src_tri]))
        dest_rect = cv2.boundingRect(np.float32([dst_tri]))

        # Get new triangle coordinates reflecting their location in bounding box
        adjusted_src_tri = [(src_tri[i][0] - src_rect[0], src_tri[i][1] - src_rect[1]) for i in range(3)]
        adjusted_dest_tri = [(dst_tri[i][0] - dest_rect[0], dst_tri[i][1] - dest_rect[1]) for i in range(3)]

        # Create mask for destination triangle
        mask = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(adjusted_dest_tri), (1.0, 1.0, 1.0), 16, 0)

        # Crop input image to corresponding bounding box
        cropped_im = im[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]

        size = (dest_rect[2], dest_rect[3])
        warpImage1 = self.affine_transform(cropped_im, adjusted_src_tri, adjusted_dest_tri, size)

        # Copy triangular region of the cropped patch to the output image
        row_start = dest_rect[1]
        row_end = dest_rect[1]+dest_rect[3]
        col_start = dest_rect[0]
        col_end = dest_rect[0]+dest_rect[2]
        im_out[row_start:row_end, col_start:col_end] = im_out[row_start:row_end, col_start:col_end] * (1 - mask) + warpImage1 * mask

    def warp_im(self, im, src_landmarks, dst_landmarks, triangles):
        # Loop over all the triangles in the image
        im_out = im.copy()
        for i in range(len(triangles)):
            src_tri = src_landmarks[triangles[i]]
            dst_tri = dst_landmarks[triangles[i]]
            self.morph_triangle(im, im_out,src_tri, dst_tri)
        return im_out

    def morph_seq(self, total_frames, im1, im2, im1_landmarks, im2_landmarks, triangles, size):
        # Create a smooth transition to morph the first image into the second
        im1 = np.float32(im1)
        im2 = np.float32(im2)

        for j in range(total_frames):
            weight = j / (total_frames - 1)
            print(f"performing face morphing for iteration {j}")
            if weight == 0.0:
                res = im1 
                # Convert to PIL Image and save to the pipe stream
                res = Image.fromarray(cv2.cvtColor(np.uint8(res), cv2.COLOR_BGR2RGB))
                cv2.imwrite(r"images/" + str(j) + ".jpg", np.array(res))
            elif weight == 1.0:
                res = im2
                # Convert to PIL Image and save to the pipe stream
                res = Image.fromarray(cv2.cvtColor(np.uint8(res), cv2.COLOR_BGR2RGB))
                cv2.imwrite(r"images/" + str(j) + ".jpg", np.array(res))
            else:
                weighted_landmarks = (1.0 - weight) * im1_landmarks + weight * im2_landmarks
                weighted_landmarks = weighted_landmarks.astype(int)
                warped_im1 = self.warp_im(im1, im1_landmarks, weighted_landmarks, triangles)
                warped_im2 = self.warp_im(im2, im2_landmarks, weighted_landmarks, triangles)

                blended = (1.0 - weight) * warped_im1 + weight * warped_im2
                res = Image.fromarray(cv2.cvtColor(np.uint8(blended), cv2.COLOR_BGR2RGB))
                cv2.imwrite(r"images/" + str(j) + ".jpg", np.array(res))
        return res

    def morph_pair(self, im1, im2, first_landmarks, second_landmarks, total_frames):
        # for a pair of images, perform face morphing
        # Triangulate the first images and copy same triangulation to second image
        triangles = self.triangulation(first_landmarks)

        h, w = im1.shape[:2]
        self.morph_seq(total_frames, im1, im2, first_landmarks, second_landmarks, triangles.tolist(), (w, h))


if __name__ == "__main__":
    predictor_path = r"models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # ask user to input the number of frames
    frames = int(input("Enter the number of frames: "))

    # Read images
    first_directory = r"images/first_img.jpeg"
    second_directory = r"images/second_img.jpg"

    first = cv2.imread(first_directory)
    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)

    second = cv2.imread(second_directory)
    second_rgb = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)

    # Ensure they are of the same size, else resize the images based on the biggest image
    # resize along the width
    if first_rgb.shape[1] > second_rgb.shape[1]:
        second_rgb = cv2.resize(second_rgb, (first_rgb.shape[1], first_rgb.shape[0]))
    else:
        first_rgb = cv2.resize(first_rgb, (second_rgb.shape[1], second_rgb.shape[0]))
    # resize along the height
    if first_rgb.shape[0] > second_rgb.shape[0]:
        second_rgb = cv2.resize(second_rgb, (first_rgb.shape[1], first_rgb.shape[0]))
    else:
        first_rgb = cv2.resize(first_rgb, (second_rgb.shape[1], second_rgb.shape[0]))

    # Get the landmarks for the first image
    FaceMorpher = FaceMorpher(predictor_path)
    first_landmarks = FaceMorpher.get_landmarks(np.copy(first_rgb), "first")
    second_landmarks = FaceMorpher.get_landmarks(np.copy(second_rgb), "second")

    # FaceMorpher.morph_pair(first_rgb, second_rgb, np.array(first_landmarks), np.array(second_landmarks), frames)

    # Create a video from the images 
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('images/face_morph.avi', fourcc, 12, (first_rgb.shape[1], first_rgb.shape[0]), isColor=True)
    for img in sorted(glob.glob(r"images/*.jpg")):
        img = cv2.imread(img)
        out.write(img)

    out.release()






