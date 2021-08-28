## create env, matching remote ray version (py38-1.6.0)
```
conda create --name ray385 python=3.8.5
conda activate ray385
pip install ray
```

## install ray cluster
Define the cluster config file, including installed dependency, mount files
```
export KUBECONFIG=xxx
ray up config/cluster.yaml
```

## create node port, expose ray server to client
```
kubectl create -f config/ray-console-service.yaml
kubectl create -f config/ray-head-service.yaml
```

## connect ray server through python or cli
Use ray.init or ray.util.connect() to connect ray server
```
ray attach config/cluster.yaml 
ray exec xxxx.yaml
ray down xxx.yaml
```