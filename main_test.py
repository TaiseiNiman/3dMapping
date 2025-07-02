import asyncio
import websockets
import json
import base64
import numpy as np
from scipy.spatial.transform import Rotation as RR
import subprocess
import os
import math  # ← 追加

async def connect_to_server():
    uri = "ws://localhost:8080/3DMapping"

    async with websockets.connect(uri, max_size=100*1024*1024) as websocket:
        print("✅ 接続しました")
        await websocket.send(json.dumps({"meta":"clientsId","contents":"compute"}))

        while True:
            try:
                message = await websocket.recv()
                contents = json.loads(message)

                for idx, item in enumerate(contents):
                    jpg_b64 = item["image"]
                    jpg_bytes = base64.b64decode(jpg_b64)
                    file_path = f"frames/{idx:04d}.jpg"
                    with open(file_path, "wb") as f:
                        f.write(jpg_bytes)

                workspace = os.getcwd()
                images_dir = os.path.join(workspace, "frames")
                database_path = os.path.join(workspace, "database.db")
                sparse_dir = os.path.join(workspace, "sparse")

                os.makedirs(sparse_dir, exist_ok=True)

                if os.path.exists(database_path):
                    os.remove(database_path)

                subprocess.run([
                    "colmap.exe", "feature_extractor",
                    "--database_path", database_path,
                    "--image_path", images_dir,
                    "--ImageReader.single_camera", "1"
                ])
                subprocess.run([
                    "colmap.exe", "exhaustive_matcher",
                    "--database_path", database_path
                ])
                subprocess.run([
                    "colmap.exe", "mapper",
                    "--database_path", database_path,
                    "--image_path", images_dir,
                    "--output_path", sparse_dir
                ])

                sparse_dir0 = os.path.join("sparse", "0")

                subprocess.run([
                    "colmap.exe", "model_converter",
                    "--input_path", sparse_dir0,
                    "--output_path", sparse_dir0,
                    "--output_type", "TXT"
                ])

                print("✅ COLMAP SfM Done.")

                sparse_subdir = sorted(os.listdir(sparse_dir))[0]
                cameras_txt = os.path.join(sparse_dir, sparse_subdir, "cameras.txt")
                images_txt = os.path.join(sparse_dir, sparse_subdir, "images.txt")

                fl_x, fl_y, cx, cy, w, h = parse_colmap_cameras(cameras_txt)

                # camera_angle_x を追加
                camera_angle_x = 2 * math.atan(0.5 * w / fl_x)

                frames = parse_colmap_images(images_txt)

                transforms = {
                    "fl_x": fl_x,
                    "fl_y": fl_y,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "camera_angle_x": camera_angle_x,  # ← 追加
                    "frames": frames
                }

                with open("transforms.json", "w") as f:
                    json.dump(transforms, f, indent=2)

                print(f"✅ transforms.json created! camera_angle_x={camera_angle_x:.5f} rad")

                await websocket.send("ping")

            except websockets.ConnectionClosed as e:
                print(f"❌ 切断されました: {e}")
                break

def parse_colmap_cameras(cameras_txt):
    with open(cameras_txt) as f:
        lines = [l.strip() for l in f if not l.startswith("#")]
    parts = lines[0].split()
    width = int(parts[2])
    height = int(parts[3])
    fl_x = float(parts[4])
    cx = float(parts[5])
    cy = float(parts[6])
    return fl_x, fl_x, cx, cy, width, height

def parse_colmap_images(images_txt):
    frames = []
    with open(images_txt) as f:
        lines = [l.strip() for l in f if not l.startswith("#")]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]

        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz]).reshape(3, 1)
        RT = np.hstack([R, t])
        RT = np.vstack([RT, [0,0,0,1]])
        transform_matrix = np.linalg.inv(RT)

        frames.append({
            "file_path": f"frames/{name}",
            "transform_matrix": transform_matrix.tolist()
        })
    return frames

def quat_to_rotmat(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz])
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        return np.identity(3)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot = np.array([
        [1.0 - q[2,2] - q[3,3], q[1,2] - q[3,0], q[1,3] + q[2,0]],
        [q[1,2] + q[3,0], 1.0 - q[1,1] - q[3,3], q[2,3] - q[1,0]],
        [q[1,3] - q[2,0], q[2,3] + q[1,0], 1.0 - q[1,1] - q[2,2]]
    ])
    return rot

if __name__ == "__main__":
    asyncio.run(connect_to_server())
