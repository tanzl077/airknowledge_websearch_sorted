import tensorflow as tf

meta_file = "bert-utils-master/tmp/result/model.ckpt-12511.meta"
checkpoint_file = "bert-utils-master/tmp/result/model.ckpt-12511"
sess = tf.Session()
imported_meta = tf.train.import_meta_graph(meta_file)
imported_meta.restore(sess, checkpoint_file)
my_vars = []
for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
        my_vars.append(var)
saver = tf.train.Saver(my_vars)
saver.save(sess, 'bert-utils-master/tmp/result/divorce_best_model.ckpt')