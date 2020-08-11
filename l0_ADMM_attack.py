## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 30  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1e-3  # the initial constant c to pick as a first guess
RO = 20

class ADMML0:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, ro = RO,
         #       initial_const=INITIAL_CONST,
                 boxmin=-0.5, boxmax=0.5):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
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
        self.boxmin = boxmin
        self.boxmax = boxmax

        self.grad = self.gradient_descent(sess, model)

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

        # the variable we're going to optimize over
        #modifier = tf.Variable(tf.random_normal(shape, stddev=1.0))
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
        # these are variables to be more efficient in sending data to tf
        timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        tlab = tf.Variable(np.zeros((batch_size, model.num_labels)), dtype=tf.float32)
        delta = tf.Variable(np.zeros(shape), dtype=tf.float32)
        multiplier = tf.Variable(np.zeros(shape), dtype=tf.float32)

        # and here's what we use to assign them
        assign_timg = tf.placeholder(tf.float32, shape)
        assign_tlab = tf.placeholder(tf.float32, (batch_size, model.num_labels))
        assign_delta = tf.placeholder(tf.float32, shape)
        assign_multiplier = tf.placeholder(tf.float32, shape)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        boxmul = (self.boxmax - self.boxmin) / 2.
        boxplus = (self.boxmin + self.boxmax) / 2.
        newimg = tf.tanh(modifier + timg) * boxmul + boxplus

        modi_real = newimg - (tf.tanh(timg) * boxmul + boxplus)

        # prediction BEFORE-SOFTMAX of the model
        output = model.predict(newimg)
        output2 = model.predict(delta+tf.tanh(timg) * boxmul + boxplus)
        # distance to the input data
        l2dist = tf.reduce_sum(tf.square(delta - modi_real + multiplier ),[1, 2, 3])

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((tlab) * output, 1)
        other = tf.reduce_max((1 - tlab) * output - (tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # sum up the losses
        loss2 = self.ro / 2.0 * l2dist
        loss1 = 8.0*loss1
        loss_batch = loss1 + loss2
        loss = tf.reduce_sum(loss_batch)

        l2dist_real = tf.reduce_sum(tf.square(modi_real), [1, 2, 3])

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        init = tf.variables_initializer(var_list=new_vars)
        init1 = tf.variables_initializer(var_list=[modifier])

        # these are the variables to initialize when we run
        setup = []
        setup.append(timg.assign(assign_timg))
        setup.append(tlab.assign(assign_tlab))

        setup2 = []
        setup2.append(delta.assign(assign_delta))
        setup2.append(multiplier.assign(assign_multiplier))

        sess.run(init1)
        def doit(imgs, labs, delt, multi):

            sess.run(init)

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            bestattack = [np.zeros(imgs[0].shape)] * batch_size

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, })

            sess.run(setup2, {assign_delta: delt, assign_multiplier: multi, })

            prev = 1e6
            bestloss = 1e10
            bestlossbatch = [1e10] * batch_size
            modi = [np.zeros(imgs[0].shape)] * batch_size
            for iteration in range(self.MAX_ITERATIONS):

                sess.run([train])
                l, lb, l2s = sess.run([loss, loss_batch, l2dist_real])
                scores, nimg = sess.run([output, newimg])
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print(iteration, l)

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * 1.09999:
                        break
                    prev = l

                modi_re = sess.run(modi_real)

                for e, (lbb, mo) in enumerate(zip(lb, modi_re)):
                    if lbb < bestlossbatch[e]:
                        bestlossbatch[e] = lbb
                        modi[e] = mo

            scores2, _ = sess.run([output2, newimg])
            for e, (sc2, iii) in enumerate(zip(scores2, delt)):
                if self.compare(sc2, np.argmax(batchlab[e])):
                    bestl2[e] = np.array(np.nonzero(iii)).shape[1]
                    bestscore[e] = sc2
                    bestattack[e] = iii

            return bestl2, bestscore, bestattack, modi

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r.extend(self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        boxmul = (self.boxmax - self.boxmin) / 2.
        boxplus = (self.boxmin + self.boxmax) / 2.
        simgs = np.arctanh((imgs - boxplus) / boxmul * 0.999999)

        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = []
        #o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        for i in range(batch_size):
            o_bestattack.append(imgs[i])

        multi = 0.0 * np.ones(imgs.shape)
        modi = 0.0 * np.ones(imgs.shape)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(outer_step, o_bestl2)

            temp = (modi - multi)

            delt = np.where(temp**2 < (2.0/self.ro), 0, temp)

            print(np.array(np.nonzero(delt)).shape[1])

            bestl2, bestscore, bestattack, modi = self.grad(simgs, labs, delt, multi)

            multi = multi + delt - modi

            for e, (l2, sc, ii) in enumerate(zip(bestl2, bestscore, bestattack)):
                if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                    o_bestl2[e] = l2
                    o_bestscore[e] = np.argmax(sc)
                    o_bestattack[e] = np.where(ii**2 < (2.0/self.ro), 0, ii)+imgs[e]

        return o_bestattack
