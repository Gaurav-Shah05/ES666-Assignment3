import numpy as np
import cv2
import glob
import os
from tqdm import tqdm

class PanaromaStitcher:
    def __init__(self):
        self.ransac_thresh = 5.0
        self.ransac_iters = 1000
        self.min_matches = 10
        self.max_image_dim = 10000  # Add max image dimension check
        
    def make_panaroma_for_images_in(self, path):
        """Create panorama from images in the given path."""
        # Load images
        all_images = sorted(glob.glob(path + os.sep + '*'))
        if len(all_images) < 2:
            raise ValueError("At least 2 images required")
            
        print(f'Found {len(all_images)} Images for stitching')
        
        images = []
        for img_path in tqdm(all_images, desc="Loading images"):
            img = cv2.imread(img_path)
            if img is not None:
                img = self.resize_if_needed(img)  # Add back resize check
                images.append(img)

        # Start with middle image
        mid = len(images) // 2
        result = images[mid]
        homography_matrix_list = []
        
        # Stitch right half
        for i in tqdm(range(mid + 1, len(images)), desc="Stitching right half"):
            warped1, warped2, H = self.stitch_pair(images[i], result)
            if warped1 is not None and warped2 is not None:
                result = self.blend_images(warped1, warped2)
                result = self.resize_if_needed(result)  # Add resize check
                homography_matrix_list.append(H)
                
        # Stitch left half
        for i in tqdm(range(mid - 1, -1, -1), desc="Stitching left half"):
            warped1, warped2, H = self.stitch_pair(images[i], result)
            if warped1 is not None and warped2 is not None:
                result = self.blend_images(warped1, warped2)
                result = self.resize_if_needed(result)  # Add resize check
                homography_matrix_list.append(H)
                
        # Final resize check
        result = self.resize_if_needed(result)
        return result, homography_matrix_list

    def resize_if_needed(self, img):
        """Resize image if it's too large."""
        max_dim = max(img.shape[0], img.shape[1])
        if max_dim > self.max_image_dim:
            scale = self.max_image_dim / max_dim
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    def validate_homography(self, H, src_shape, dst_shape):
        """Validate homography matrix and resulting image size."""
        if H is None:
            return False
            
        # Check condition number
        if np.linalg.cond(H) > 1e15:
            return False
            
        # Check resulting image size
        corners = np.float32([[0, 0], [0, src_shape[0]], 
                            [src_shape[1], src_shape[0]], [src_shape[1], 0]])
        transformed_corners = self.perspective_transform(corners.reshape(-1, 1, 2), H)
        min_x, min_y = transformed_corners.reshape(-1, 2).min(axis=0)
        max_x, max_y = transformed_corners.reshape(-1, 2).max(axis=0)
        
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        
        # Check if resulting image would be too large
        if width > self.max_image_dim or height > self.max_image_dim:
            return False
            
        return True

    def detect_and_match_features(self, img1, img2):
        """Match features between two images using SIFT."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT features
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)
        
        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return src_pts, dst_pts

    def compute_homography_ransac(self, src_pts, dst_pts):
        """Compute homography matrix using RANSAC."""
        if len(src_pts) < 4:
            return None
            
        best_H = None
        max_inliers = 0
        
        for _ in range(self.ransac_iters):
            # Randomly select 4 points
            idx = np.random.choice(len(src_pts), 4, replace=False)
            H = self.compute_homography_dlt(src_pts[idx], dst_pts[idx])
            
            if H is None:
                continue
                
            # Count inliers
            inliers = self.count_inliers(H, src_pts, dst_pts)
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H
        
        return best_H if max_inliers >= self.min_matches else None

    def compute_homography_dlt(self, src_pts, dst_pts):
        """Compute homography using Direct Linear Transform."""
        if len(src_pts) != 4 or len(dst_pts) != 4:
            return None
            
        A = np.zeros((8, 9))
        
        for i in range(4):
            x, y = src_pts[i]
            u, v = dst_pts[i]
            A[i*2] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
            A[i*2 + 1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]
            
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]  # Normalize
        return H

    def count_inliers(self, H, src_pts, dst_pts):
        """Count number of inliers for a given homography."""
        src_homog = np.column_stack([src_pts, np.ones(len(src_pts))])
        transformed = (H @ src_homog.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]
        distances = np.linalg.norm(transformed - dst_pts, axis=1)
        return np.sum(distances < self.ransac_thresh)

    def warp_image(self, img, H, output_shape):
        """Warp image using homography matrix."""
        height, width = output_shape
        warped = np.zeros((height, width, 3), dtype=np.uint8)
        H_inv = np.linalg.inv(H)
        
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        
        transformed = coords @ H_inv.T
        transformed = transformed[..., :2] / transformed[..., 2:]
        
        x_transformed = transformed[..., 0].astype(np.int32)
        y_transformed = transformed[..., 1].astype(np.int32)
        
        mask = (x_transformed >= 0) & (x_transformed < img.shape[1]) & \
               (y_transformed >= 0) & (y_transformed < img.shape[0])
        
        warped[y_coords[mask], x_coords[mask]] = \
            img[y_transformed[mask], x_transformed[mask]]
            
        return warped

    def blend_images(self, img1, img2):
        """Blend two images using alpha blending."""
        mask1 = (img1 != 0).any(axis=2)
        mask2 = (img2 != 0).any(axis=2)
        overlap = mask1 & mask2
        
        result = np.zeros_like(img1)
        result[mask1] = img1[mask1]
        result[mask2] = img2[mask2]
        
        # Alpha blend in overlap region
        x_coords = np.linspace(0, 1, img1.shape[1])
        alpha = np.tile(x_coords, (img1.shape[0], 1))
        result[overlap] = (alpha[overlap, np.newaxis] * img1[overlap] + 
                         (1 - alpha[overlap, np.newaxis]) * img2[overlap])
        
        return result

    def stitch_pair(self, img1, img2):
        """Stitch a pair of images."""
        # Match features
        src_pts, dst_pts = self.detect_and_match_features(img1, img2)
        
        if len(src_pts) < 4:
            return None, None, None
            
        # Compute homography
        H = self.compute_homography_ransac(src_pts, dst_pts)
        if H is None or not self.validate_homography(H, img1.shape[:2], img2.shape[:2]):
            return None, None, None
            
        # Calculate output size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        
        # Transform corners and get bounds
        corners1_transformed = self.perspective_transform(corners1, H)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        all_corners = np.vstack([corners2.reshape(-1, 2), corners1_transformed.reshape(-1, 2)])
        
        [xmin, ymin] = np.int32(np.floor(all_corners.min(axis=0)))
        [xmax, ymax] = np.int32(np.ceil(all_corners.max(axis=0)))
        
        # Check output size
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        if width > self.max_image_dim or height > self.max_image_dim:
            return None, None, None
        
        # Adjust homography
        translation = np.array([[1, 0, -xmin],
                              [0, 1, -ymin],
                              [0, 0, 1]])
        H_adjusted = translation @ H
        
        # Warp and create output images
        output_shape = (height, width)
        warped1 = self.warp_image(img1, H_adjusted, output_shape)
        warped2 = np.zeros_like(warped1)
        warped2[-ymin:-ymin+h2, -xmin:-xmin+w2] = img2
        
        return warped1, warped2, H

    def perspective_transform(self, points, H):
        """Apply perspective transform to points."""
        if len(points.shape) == 3:
            points = points.reshape(-1, 2)
            
        homog_pts = np.column_stack([points, np.ones(len(points))])
        transformed = homog_pts @ H.T
        transformed = transformed[:, :2] / transformed[:, 2:]
        
        return transformed.reshape(-1, 1, 2) if len(points.shape) == 3 else transformed
