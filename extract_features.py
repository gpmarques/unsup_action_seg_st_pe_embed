import subprocess
import os
import runpy

root = './BreakfastII_15fps_qvga_sync'
video_paths = './video_paths.txt'
SEGMENT_SIZE = 128

for walk in os.walk(root):
    for folder in walk[1]:
        partition = os.path.join(root, folder)
        for sub_folder in os.listdir(partition):
            print(sub_folder)
            cam = os.path.join(partition, sub_folder)
            cam_features_path = os.path.join(cam, f'i3d_features_{SEGMENT_SIZE}frames')
            videos = [file for file in os.listdir(cam) if file.split(".")[-1] == 'avi']
            
            if len(videos) > 0:
                if os.path.exists(video_paths):
                    os.remove(video_paths)
                
                with open(video_paths, 'w') as paths_file:
                    for video in videos:
                        paths_file.write(os.path.join(cam, video)+"\n")
                
                for video in videos:
                    comps = video.split("_")[:2]
                    comps[1] = comps[1].split(".")[0]
                    split_v = "_".join(comps)
                    f_path = os.path.join(cam_features_path, 'i3d', split_v + "_flow.npy")
                    if os.path.exists(f_path):
                        pass
                    else:
                        print(f_path)
                        command = f'python main.py --feature_type i3d --device_ids 0 --stack_size {SEGMENT_SIZE} --step_size {SEGMENT_SIZE} --output_path {cam_features_path} --on_extraction save_numpy --video_paths {video_paths}'
                        print(command)
                        print(subprocess.call([command], shell=True))

            else:
                print(f'{cam} is empty')
    break
