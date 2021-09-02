import numpy as np
import bundle
import time
import ray
from tqdm import tqdm

ray.init("auto")

@ray.remote
def ray_bundles(idx):
    bundle_list, Z_list = ray.get(ray_env).create_bundles(idx,idx+1)
    return  bundle_list,Z_list
t1 = time.time()
env = bundle.Evaluator()
ray_env = ray.put(env)
return_refs = [ray_bundles.remote(idx) for idx in range(260, 265)]
return_items = ray.get(return_refs)
print(f"Total time = {time.time() - t1}")

B_items = [return_items[i][0][0] for i in range(len(return_items))]
Z_items = [return_items[i][1][0] for i in range(len(return_items))]
import pickle
B_file = "./Data/B_list.pkl"
Z_file = "./Data/Z_list.pkl"
with open(B_file, 'wb') as f:
    pickle.dump(B_items, f)
with open(Z_file, 'wb') as f:
    pickle.dump(Z_items, f)