import numpy as np
import cv2
import sys
from matchers import SIFTMatcher
import time
import blend
import matplotlib.pyplot as plt

class Stitch:
    def __init__(self, args):
        self.path = args
        fp = open(self.path, 'r')
        filenames = [each.rstrip('\r\n') for each in fp.readlines()]
        # filenames = args
        print(filenames)
        # self.images = [cv2.resize(cv2.imread(each), (480, 320)) for each in filenames]
        self.images = [cv2.imread(each) for each in filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = SIFTMatcher()
        self.performance_metrics = {
            'keypoints': [],
            'matches': [],
            'match_distances': [],
            'homography_dets': [],
            'processing_times': []
        }
        self.prepare_lists()

    def prepare_lists(self):
        print("Number of images : %d" % self.count)
        self.centerIdx = self.count / 2
        print("Center index image : %d" % self.centerIdx)
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if (i <= self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print("Image lists prepared")

    def leftshift(self):
        # self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        for b in self.left_list[1:]:
            start_time = time.time()
            result = self.matcher_obj.match(a, b, 'left', return_metrics=True)
            processing_time = time.time() - start_time

            self.performance_metrics['matches'].append(result['num_matches'])
            self.performance_metrics['match_distances'].append(result['match_distances'])
            self.performance_metrics['keypoints'].extend([result['keypoints1'], result['keypoints2']])
            self.performance_metrics['homography_dets'].append(result['homography_det'])
            self.performance_metrics['processing_times'].append(processing_time)

            H = result['H']
            # print("Homography is : ", H)
            xh = np.linalg.inv(H)
            # print("Inverse Homography :", xh)
            br = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            br = br /br[-1]
            tl = np.dot(xh, np.array([0, 0, 1]))
            tl = tl / tl[-1]
            bl = np.dot(xh, np.array([0, a.shape[0], 1]))
            bl = bl / bl[-1]
            tr = np.dot(xh, np.array([a.shape[1], 0, 1]))
            tr = tr / tr[-1]
            cx = int(max([0, a.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, a.shape[0], tl[1], bl[1], tr[1], br[1]]))
            offset = [abs(int(min([0, a.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, a.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            tl[0:2] += offset; bl[0:2] += offset;  tr[0:2] += offset; br[0:2] += offset
            dstpoints = np.array([tl, bl, tr, br]);
            srcpoints = np.array([[0, 0], [0, a.shape[0]], [a.shape[1], 0], [a.shape[1], a.shape[0]]])
            # print('sp',sp,'dp',dp)
            M_off = cv2.findHomography(srcpoints, dstpoints)[0]
            # print('M_off', M_off)
            warped_img2 = cv2.warpPerspective(a, M_off, dsize)
            # cv2.imshow("warped", warped_img2)
            # cv2.waitKey()
            warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img1[offset[1]:b.shape[0] + offset[1], offset[0]:b.shape[1] + offset[0]] = b
            tmp = blend.blend_linear(warped_img1, warped_img2)
            a = tmp

        self.leftImage = tmp

    def rightshift(self):
        tmp = self.leftImage  # Initialize with left image
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            # print("Homography :", H)
            br = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            br = br / br[-1]
            tl = np.dot(H, np.array([0, 0, 1]))
            tl = tl / tl[-1]
            bl = np.dot(H, np.array([0, each.shape[0], 1]))
            bl = bl / bl[-1]
            tr = np.dot(H, np.array([each.shape[1], 0, 1]))
            tr = tr / tr[-1]
            cx = int(max([0, self.leftImage.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, self.leftImage.shape[0], tl[1], bl[1], tr[1], br[1]]))
            offset = [abs(int(min([0, self.leftImage.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, self.leftImage.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            tl[0:2] += offset; bl[0:2] += offset; tr[0:2] += offset; br[0:2] += offset
            dstpoints = np.array([tl, bl, tr, br]);
            srcpoints = np.array([[0, 0], [0, each.shape[0]], [each.shape[1], 0], [each.shape[1], each.shape[0]]])
            M_off = cv2.findHomography(dstpoints, srcpoints)[0]
            warped_img2 = cv2.warpPerspective(each, M_off, dsize, flags=cv2.WARP_INVERSE_MAP)
            # cv2.imshow("warped", warped_img2)
            # cv2.waitKey()
            warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img1[offset[1]:self.leftImage.shape[0] + offset[1], offset[0]:self.leftImage.shape[1] + offset[0]] = self.leftImage
            tmp = blend.blend_linear(warped_img1, warped_img2)
            self.leftImage = tmp
            
        self.rightImage = tmp
    
    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()

    def evaluate_performance(self):
        """Generate comprehensive performance analysis visualizations"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 18))
        gs = fig.add_gridspec(3, 2)
        axes = [
            fig.add_subplot(gs[0, 0]),  # Keypoints histogram
            fig.add_subplot(gs[0, 1]),  # Matches bar chart
            fig.add_subplot(gs[1, 0]),  # Match distances boxplot
            fig.add_subplot(gs[1, 1]),  # Homography determinants
            fig.add_subplot(gs[2, :])   # Processing times
        ]

        # 1. Keypoints Distribution
        kp_counts = [len(kp) for kp in self.performance_metrics['keypoints']]
        axes[0].hist(kp_counts, bins=20, color='teal', alpha=0.7)
        axes[0].set_title('Keypoints Distribution per Image')
        axes[0].set_xlabel('Number of Keypoints')
        axes[0].set_ylabel('Frequency')

        # 2. Feature Matches Analysis
        pair_labels = [f'{i}-{i+1}' for i in range(len(self.performance_metrics['matches']))]
        axes[1].bar(pair_labels, self.performance_metrics['matches'], color='skyblue')
        axes[1].set_title('Feature Matches Between Consecutive Pairs')
        axes[1].set_ylabel('Number of Matches')
        axes[1].tick_params(axis='x', rotation=45)

        # 3. Match Quality Boxplot
        all_distances = [d for sublist in self.performance_metrics['match_distances'] for d in sublist]
        axes[2].boxplot(self.performance_metrics['match_distances'], showfliers=False)
        axes[2].set_title('Match Distance Distribution (Lower = Better)')
        axes[2].set_ylabel('L2 Distance')
        axes[2].set_xticklabels(pair_labels)

        # 4. Homography Matrix Determinants
        dets = np.abs(self.performance_metrics['homography_dets'])
        axes[3].plot(dets, marker='o', linestyle='--', color='purple')
        axes[3].set_title('Homography Determinants (Scale Factors)')
        axes[3].set_ylabel('|det(H)|')
        axes[3].axhline(1, color='red', linestyle=':')
        axes[3].set_xticks(range(len(dets)))
        axes[3].set_xticklabels(pair_labels)

        # 5. Processing Time Analysis
        times = self.performance_metrics['processing_times']
        axes[4].plot(times, marker='o', color='orange')
        axes[4].set_title('Processing Time per Image Pair')
        axes[4].set_ylabel('Time (seconds)')
        axes[4].set_xlabel('Image Pair Index')
        axes[4].set_xticks(range(len(times)))
        axes[4].set_xticklabels(pair_labels)

        plt.tight_layout()
        plt.savefig('results/performance_dashboard.png')
        plt.close()

if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        args = "txtlists/files2.txt"
    finally:
        print("Parameters : ", args)
    s = Stitch(args)
    
    # images = ['images/S1.jpg', 'images/S2.jpg','images/S3.jpg','images/S5.jpg','images/S6.jpg']
    # images = ['images/trees_00{}Hill.jpg'.format(i) for i in range(0, 4)]
    # s = Stitch(images)

    s.leftshift()
    # s.showImage('left')
    s.rightshift()
    print("done")
    cv2.imwrite("results/test2.jpg", s.leftImage)
    print("image written")
    cv2.destroyAllWindows()
    s.evaluate_performance()
        
