
import sys
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from scipy import io

BINARY_SEARCH_STEPS = 30  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
RO = 20
u = 0.5
Query_iterations = 20
GAMA = 1
EPI = 1


class LADMMBB:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, ro=RO, gama=GAMA, epi=EPI):
        """
        The L_2 optimized attack.
        """

        self.model = model
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.ro = ro
        self.grad = self.gradient_descent(sess, model)
        self.grad2 = self.gradient_descent2(sess, model)
        self.gama = gama
        self.epi=epi

    def compare(self, x, y):
        if not isinstance(x, (float, int, np.int64)):
            x = np.copy(x)
            if self.TARGETED:
                x[y] -= self.CONFIDENCE
            else:
                x[y] += self.CONFIDENCE
            x = np.argmax(x)
        if self.TARGETED:
            return x == y
        else:
            return x != y

    def gradient_descent(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        tz = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_tz = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        newimg = tz + timg
        l2dist_real = tf.reduce_sum(tf.square(tz), [1, 2, 3])
        output = model.predict(newimg)

        real = tf.reduce_sum(tlab * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

 #       loss1 = tf.reduce_sum(loss1)

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(tz.assign(assign_tz))

        def doit(imgs, labs, z):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tz: z,})

            l2s, scores, nimg, loss = sess.run([l2dist_real, output, newimg, loss1])

            return l2s, scores, nimg, loss

        return doit

    def gradient_descent2(self, sess, model):

        batch_size = self.batch_size
        shape = (batch_size, model.image_size, model.image_size, model.num_channels)

        tz = tf.Variable(np.zeros(shape, dtype=np.float32))
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_tz = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        newimg = tz + timg
        l2dist_real = tf.reduce_sum(tf.square(tz), [1, 2, 3])
        output = model.predict(newimg)

        real = tf.reduce_sum(tlab * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))
        setup.append(tz.assign(assign_tz))

        def doit(imgs, labs, z):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tz: z,})

            loss = sess.run([loss1])

            return loss

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.
        """
        r = []
        qc = []
        ql2 = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            attac, queryc, queryl2 = self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size])
            r.extend(attac)
            qc.extend(queryc)
            ql2.extend(queryl2)
        return np.array(r), np.array(qc), np.array(ql2)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [imgs[0]] * batch_size
        o_querycount = [1e10] * batch_size
        o_queryl2 = [1e10] * batch_size
        o_conv = [1e10] * self.BINARY_SEARCH_STEPS

        delt = 0.0 * np.ones(imgs.shape)
        s = 0.0 * np.ones(imgs.shape)
        alpha = 10

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_step, o_bestl2)

            tempa = delt - (1/self.ro) * s

            ztemp = self.ro/(self.ro + 2 * self.gama) * tempa
            mintemp = np.where( 0.5-imgs < self.epi,  0.5-imgs, self.epi)
            maxtemp = np.where( -0.5-imgs > -self.epi, -0.5-imgs, -self.epi)
            ztemp1 = np.where( ztemp > mintemp, mintemp, ztemp )
            z = np.where( ztemp1 < maxtemp, maxtemp, ztemp1 )

            loss = self.grad2(imgs, labs, delt)

            delt_grads = np.zeros(imgs.shape)

            for iii in range(Query_iterations):
                dirc = np.random.normal(0, 1, imgs.shape)
                sss = LA.norm(dirc.reshape(batch_size,-1), axis=1)
                sss = sss.reshape(batch_size, 1, 1, 1)
                dir_normal = dirc / sss
                temploss = self.grad2( imgs + u * dir_normal, labs, delt)
                delt_grads = delt_grads + (temploss[0] - loss[0]).reshape(batch_size, 1, 1, 1)*dir_normal

            delt_grads = delt_grads*imgs.shape[1]*imgs.shape[2]*imgs.shape[3]/(u * Query_iterations)

            eta = np.sqrt(outer_step+1)
            delt = 1/(alpha * eta + self.ro) * \
                    (alpha * eta * delt + self.ro * z + s - delt_grads)

            l2s, scores, nimg, loss1 = self.grad(imgs, labs, delt)
            print(loss1[0])
            s = s + self.ro * ( z - delt )

            o_conv[outer_step] = np.sqrt(l2s[0])
            for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
                        if o_querycount[e] == 1e10:
                            o_querycount[e] = (outer_step + 1) * ( Query_iterations + 1)
                            o_queryl2[e] = l2

        file = open('conv.txt', 'w')
        file.write(str(o_conv))
        file.close()
        return o_bestattack, o_querycount, o_queryl2
