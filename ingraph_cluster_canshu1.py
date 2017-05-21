import tensorflow as tf

cluster=tf.train.ClusterSpec({
    "canshu": [
        "172.16.100.2:12222",# /job:canshu/task:0 运行的主机
        "172.16.100.3:12222",# /job:canshu/task:1 运行的主机
    ],
    "gongzuo": [
        "172.16.100.4:12222",  # /job:gongzuo/task:0 运行的主机
        "172.16.100.5:12222"   # /job:gongzuo/task:1 运行的主机
    ]})

server = tf.train.Server(cluster, job_name="canshu", task_index=1)

server.join()
