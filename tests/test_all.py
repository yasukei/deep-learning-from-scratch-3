import unittest

import numpy as np
import chainer
import chainer.functions as CF

from dezero.all import *

# =============================================================================
# =============================================================================
# =============================================================================
#
# test_basic_math.py
#
# =============================================================================
# =============================================================================
# =============================================================================

class TestAdd(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 + x1
        res = y.data
        expected = np.array([2, 4, 6])
        self.assertTrue(array_equal(res, expected))

    def test_datatype(self):
        """np.float64ではなく、0次元のndarrayを返すかどうか"""
        x = Variable(np.array(2.0))
        y = x ** 2
        self.assertFalse(np.isscalar(y))

    def test_backward1(self):
        x = Variable(np.random.randn(3, 3))
        y = np.random.randn(3, 3)
        f = lambda x: x + y
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = Variable(np.random.randn(3, 3))
        y = np.random.randn(3, 1)
        f = lambda x: x + y
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        self.assertTrue(gradient_check(add, x, y))


class TestMul(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 * x1
        res = y.data
        expected = np.array([1, 4, 9])
        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x: x * y
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x * y
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda y: x * y
        self.assertTrue(gradient_check(f, x))


class TestDiv(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 / x1
        res = y.data
        expected = np.array([1, 1, 1])
        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x: x / y
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x / y
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x: x / y
        self.assertTrue(gradient_check(f, x))

# =============================================================================
# =============================================================================
# =============================================================================
#
# test_batchnorm.py
#
# =============================================================================
# =============================================================================
# =============================================================================
def get_params(N, C, H=None, W=None, dtype='f'):
    if H is not None:
        x = np.random.randn(N, C, H, W).astype(dtype)
    else:
        x = np.random.randn(N, C).astype(dtype)
    gamma = np.random.randn(C).astype(dtype)
    beta = np.random.randn(C).astype(dtype)
    mean = np.random.randn(C).astype(dtype)
    var = np.abs(np.random.randn(C).astype(dtype))
    return x, gamma, beta, mean, var


class TestFixedBatchNorm(unittest.TestCase):

    def test_type1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        with test_mode():
            y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        N, C = 8, 1
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with test_mode():
            y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward2(self):
        N, C = 1, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with test_mode():
            y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward3(self):
        N, C = 20, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with test_mode():
            y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward4(self):
        N, C, H, W = 20, 10, 5, 5
        x, gamma, beta, mean, var = get_params(N, C, H, W)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with test_mode():
            y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))




class TestBatchNorm(unittest.TestCase):

    def test_type1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        N, C = 8, 1
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.batch_normalization(x, gamma, beta, running_mean=mean, running_var=var)
        y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward2(self):
        N, C = 1, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.batch_normalization(x, gamma, beta)
        y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward3(self):
        N, C = 20, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.batch_normalization(x, gamma, beta)
        y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward4(self):
        N, C, H, W = 20, 10, 5, 5
        x, gamma, beta, mean, var = get_params(N, C, H, W)
        cy = CF.batch_normalization(x, gamma, beta)
        y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward5(self):
        N, C = 20, 10
        cl = chainer.links.BatchNormalization(C)
        l = L_BatchNorm()

        for i in range(10):
            x = np.random.randn(N, C).astype('f')
            cy = cl(x)
            y = l(x)
            self.assertTrue(array_allclose(y.data, cy.data))
        self.assertTrue(array_allclose(cl.avg_mean.data, l.avg_mean.data))
        self.assertTrue(array_allclose(cl.avg_var.data, l.avg_var.data))

    def test_forward6(self):
        N, C, H, W = 20, 10, 5, 5
        cl = chainer.links.BatchNormalization(C)
        l = L_BatchNorm()

        for i in range(10):
            x = np.random.randn(N, C, H, W).astype('f')
            cy = cl(x)
            y = l(x)
            self.assertTrue(array_allclose(y.data, cy.data))
        self.assertTrue(array_allclose(cl.avg_mean.data, l.avg_mean.data))
        self.assertTrue(array_allclose(cl.avg_var.data, l.avg_var.data))


    def test_backward1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
        f = lambda x: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
        f = lambda gamma: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, gamma))

    def test_backward3(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
        f = lambda beta: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, beta))

    def test_backward4(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
        f = lambda x: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, x))

    def test_backward5(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
        f = lambda gamma: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, gamma))

    def test_backward6(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
        f = lambda beta: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, beta))


class TestBatchNormLayer(unittest.TestCase):

    def test_forward1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        cy = chainer.links.BatchNormalization(3)(x)
        y = L_BatchNorm()(x)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward2(self):
        N, C = 8, 3
        cl = chainer.links.BatchNormalization(C)
        l = L_BatchNorm()
        for i in range(10):
            x, gamma, beta, mean, var = get_params(N, C)
            cy = cl(x)
            y = l(x)
        self.assertTrue(array_allclose(cl.avg_mean, l.avg_mean.data))
        self.assertTrue(array_allclose(cl.avg_var, l.avg_var.data))

        with test_mode():
            y = l(x)
        with chainer.using_config('train', False):
            cy = cl(x)
        self.assertTrue(array_allclose(cy.data, y.data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_broadcast.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestBroadcast(unittest.TestCase):

    def test_shape_check(self):
        x = Variable(np.random.randn(1, 10))
        b = Variable(np.random.randn(10))
        y = x + b
        loss = F_sum(y)
        loss.backward()
        self.assertEqual(b.grad.shape, b.shape)


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_conv2d.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestConv2d_simple(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (1, 1), (1, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (3, 1), (2, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_forward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_forward4(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = np.random.randn(o).astype('f')
        y = conv2d_simple(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda x: conv2d_simple(x, W, b, s, p)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda b: conv2d_simple(x, W, b, s, p)
        self.assertTrue(gradient_check(f, b))

    def test_backward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda W: conv2d_simple(x, W, b, s, p)
        self.assertTrue(gradient_check(f, W))


class TestConv2d(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (1, 1), (1, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (3, 1), (2, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_forward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_forward4(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = np.random.randn(o).astype('f')
        y = conv2d(x, W, b, s, p)
        expected = CF.convolution_2d(x, W, b, s, p)
        self.assertTrue(array_equal(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda x: conv2d(x, W, b, s, p)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda b: conv2d(x, W, b, s, p)
        self.assertTrue(gradient_check(f, b))

    def test_backward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w)
        W = np.random.randn(o, c, k[0], k[1])
        b = np.random.randn(o)
        f = lambda W: conv2d(x, W, b, s, p)
        self.assertTrue(gradient_check(f, W))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_deconv2d.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestDeconv2d(unittest.TestCase):

    def test_forward1(self):
        n, c_i, c_o = 10, 1, 3
        h_i, w_i = 5, 10
        h_k, w_k = 10, 10
        h_p, w_p = 5, 5
        s_y, s_x = 5, 5
        x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)
        W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k)).astype(np.float32)
        b = np.random.uniform(0, 1, c_o).astype(np.float32)

        expected = CF.deconvolution_2d(x, W, b, stride=(s_y, s_x),
                                       pad=(h_p, w_p))
        y = deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c_i, c_o = 10, 1, 3
        h_i, w_i = 5, 10
        h_k, w_k = 10, 10
        h_p, w_p = 5, 5
        s_y, s_x = 5, 5
        x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)
        W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k)).astype(np.float32)
        b = None
        expected = CF.deconvolution_2d(x, W, b, stride=(s_y, s_x),
                                       pad=(h_p, w_p))
        y = deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c_i, c_o = 10, 1, 3
        h_i, w_i = 5, 10
        h_k, w_k = 10, 10
        h_p, w_p = 5, 5
        s_y, s_x = 5, 5
        x = np.random.uniform(0, 1, (n, c_i, h_i, w_i))
        W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k))
        b = None  # np.random.uniform(0, 1, c_o).astype(np.float32)
        f = lambda x: deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c_i, c_o = 10, 1, 3
        h_i, w_i = 5, 10
        h_k, w_k = 10, 10
        h_p, w_p = 5, 5
        s_y, s_x = 5, 5
        x = np.random.uniform(0, 1, (n, c_i, h_i, w_i))
        W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k))
        b = np.random.uniform(0, 1, c_o)
        f = lambda W: deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        self.assertTrue(gradient_check(f, W))

    def test_backward3(self):
        n, c_i, c_o = 10, 1, 3
        h_i, w_i = 5, 10
        h_k, w_k = 10, 10
        h_p, w_p = 5, 5
        s_y, s_x = 5, 5
        x = np.random.uniform(0, 1, (n, c_i, h_i, w_i))
        W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k))
        b = np.random.uniform(0, 1, c_o)
        f = lambda b: deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        self.assertTrue(gradient_check(f, b))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_dropout.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestDropout(unittest.TestCase):

    def test_forward1(self):
        x = np.random.randn(100, 100)
        y = dropout(Variable(x), dropout_ratio=0.0)
        res = array_equal(y.data, x)
        self.assertTrue(res)

    def test_forward2(self):
        x = np.random.randn(100, 100)
        with test_mode():
            y = dropout(x)
        res = array_equal(y.data, x)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.random.randn(10, 10)

        def f(x):
            np.random.seed(0)
            return dropout(x, 0.5)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 20)

        def f(x):
            np.random.seed(0)
            return dropout(x, 0.99)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 10)

        def f(x):
            np.random.seed(0)
            return dropout(x, 0.0)

        self.assertTrue(gradient_check(f, x_data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_getitem.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestGetitem(unittest.TestCase):

    def test_forward1(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = get_item(x, 0)
        self.assertTrue(array_allclose(y.data, x_data[0]))

    def test_forward1a(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = x[0]
        self.assertTrue(array_allclose(y.data, x_data[0]))

    def test_forward2(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = get_item(x, (0, 0, slice(0, 2, 1)))
        self.assertTrue(array_allclose(y.data, x_data[0, 0, 0:2:1]))

    def test_forward3(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = get_item(x, (Ellipsis, 2))
        self.assertTrue(array_allclose(y.data, x_data[..., 2]))

    def test_backward1(self):
        x_data = np.array([[1, 2, 3], [4, 5, 6]])
        slices = 1
        f = lambda x: get_item(x, slices)
        gradient_check(f, x_data)

    def test_backward2(self):
        x_data = np.arange(12).reshape(4, 3)
        slices = slice(1, 3)
        f = lambda x: get_item(x, slices)
        gradient_check(f, x_data)


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_im2col.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestIm2col(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.arange(n * c * h * w).reshape((n, c, h, w))
        y = im2col(x, 3, 3, 0, to_matrix=True)
        expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])

        res = array_equal(y.data, expected)
        self.assertTrue(res)

    def test_backward1(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.arange(n * c * h * w).reshape((n, c, h, w))
        f = lambda x: im2col(x, 3, 3, 0, to_matrix=True)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.arange(n * c * h * w).reshape((n, c, h, w))
        f = lambda x: im2col(x, 3, 3, 0, to_matrix=False)
        self.assertTrue(gradient_check(f, x))


class TestCol2in(unittest.TestCase):

    def test_backward1(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.random.rand(1, 9)
        f = lambda x: col2im(x, (n, c, h, w), 3, 3, 0, to_matrix=True)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.random.rand(1, 1, 3, 3, 1, 1)
        f = lambda x: col2im(x, (n, c, h, w), 3, 3, 0, to_matrix=False)
        self.assertTrue(gradient_check(f, x))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_linear.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestLinear(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        b = None
        y = linear(x, w, b)

        res = y.data
        expected = np.array([[14, 32], [32, 77]])
        self.assertTrue(array_allclose(res, expected))

    def test_forward2(self):
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype('f')
        W = x.T
        b = None
        y = linear(x, W, b)

        cy = chainer.functions.linear(x, W.T)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward3(self):
        layer = chainer.links.Linear(3, 2)
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype('f')
        W = layer.W.data.T
        b = layer.b.data
        y = linear(x, W, b)

        cy = layer(x)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        W = np.random.randn(2, 3)
        b = np.random.randn(3)
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        W = np.random.randn(2, 3)
        b = np.random.randn(3)
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(100, 200)
        W = np.random.randn(200, 300)
        b = None
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_loss.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestMSE_simple(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([0.0, 1.0, 2.0])
        x1 = np.array([0.0, 1.0, 2.0])
        expected = ((x0 - x1) ** 2).sum() / x0.size
        y = mean_squared_error_simple(x0, x1)
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x0 = np.random.rand(10)
        x1 = np.random.rand(10)
        f = lambda x0: mean_squared_error_simple(x0, x1)
        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.rand(100)
        x1 = np.random.rand(100)
        f = lambda x0: mean_squared_error_simple(x0, x1)
        self.assertTrue(gradient_check(f, x0))


class TestMSE_simple(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([0.0, 1.0, 2.0])
        x1 = np.array([0.0, 1.0, 2.0])
        expected = ((x0 - x1) ** 2).sum() / x0.size
        y = mean_squared_error(x0, x1)
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x0 = np.random.rand(10)
        x1 = np.random.rand(10)
        f = lambda x0: mean_squared_error(x0, x1)
        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.rand(100)
        x1 = np.random.rand(100)
        f = lambda x0: mean_squared_error(x0, x1)
        self.assertTrue(gradient_check(f, x0))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_matmul.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestMatmul(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        y = matmul(x, w)
        res = y.data
        expected = np.array([[14, 32], [32, 77]])
        self.assertTrue(array_allclose(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        w = np.random.randn(2, 3)
        f = lambda x: matmul(x, Variable(w))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x_data = np.random.randn(10, 1)
        w_data = np.random.randn(1, 5)
        f = lambda w: matmul(Variable(x_data), w)
        self.assertTrue(gradient_check(f, w_data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_max.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestMax(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.rand(10))
        y = F_max(x)
        expected = np.max(x.data)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward2(self):
        shape = (10, 20, 30)
        axis = 1
        x = Variable(np.random.rand(*shape))
        y = F_max(x, axis=axis)
        expected = np.max(x.data, axis=axis)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward3(self):
        shape = (10, 20, 30)
        axis = (0, 1)
        x = Variable(np.random.rand(*shape))
        y = F_max(x, axis=axis)
        expected = np.max(x.data, axis=axis)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward4(self):
        shape = (10, 20, 30)
        axis = (0, 1)
        x = Variable(np.random.rand(*shape))
        y = F_max(x, axis=axis, keepdims=True)
        expected = np.max(x.data, axis=axis, keepdims=True)
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x_data = np.random.rand(10)
        f = lambda x: F_max(x)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10) * 100
        f = lambda x: F_max(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 30) * 100
        f = lambda x: F_max(x, axis=(1, 2))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.rand(10, 20, 20) * 100
        f = lambda x: F_sum(x, axis=None)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward5(self):
        x_data = np.random.rand(10, 20, 20) * 100
        f = lambda x: F_sum(x, axis=None, keepdims=True)
        self.assertTrue(gradient_check(f, x_data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_pooling.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestPooling_simple(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = pooling_simple(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = pooling_simple(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad, cover_all=False)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 100
        f = lambda x: pooling_simple(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))


class TestPooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad, cover_all=False)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 1000
        f = lambda x: pooling(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))


class TestAveragePooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = average_pooling(x, ksize, stride, pad)
        expected = CF.average_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = average_pooling(x, ksize, stride, pad)
        expected = CF.average_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 1000
        f = lambda x: average_pooling(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_relu.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestRelu(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        res = relu(x)
        ans = np.array([[0, 0], [2, 0], [0, 1]], np.float32)
        self.assertTrue(array_allclose(res, ans))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])
        self.assertTrue(gradient_check(relu, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10) * 100
        self.assertTrue(gradient_check(relu, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10) * 100
        self.assertTrue(gradient_check(relu, x_data))


class TestLeakyRelu(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        res = leaky_relu(x)
        ans = np.array([[-0.2, 0.], [2., -0.6], [-0.4, 1.]], np.float32)
        self.assertTrue(array_allclose(res, ans))

    def test_forward2(self):
        slope = 0.002
        x = np.random.randn(100)
        y2 = CF.leaky_relu(x, slope)
        y = leaky_relu(x, slope)
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])
        self.assertTrue(gradient_check(leaky_relu, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10) * 100
        self.assertTrue(gradient_check(leaky_relu, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10) * 100
        self.assertTrue(gradient_check(leaky_relu, x_data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_resnet.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestResnet152(unittest.TestCase):

    def test_forward1(self):
        x = np.random.randn(1, 3, 224, 224).astype('f')
        _model = chainer.links.ResNet152Layers(None)

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                out_layer_name = 'fc6'
                _y = _model.forward(x, [out_layer_name])[out_layer_name]

        print(_y.shape)
        """
        model = VGG16()
        layers = _model.available_layers
        for l in layers:
            if "conv" in l or "fc" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                m1.W.data = m2.W.data
                m1.b.data = m2.b.data
                if "fc" in l:
                    m1.W.data = m1.W.data.T

        with test_mode():
            y = model(x)
        
        self.assertTrue(array_allclose(y.data, _y.data))
        """


    def test_forward2(self):
        x = np.random.randn(1, 3, 224, 224).astype('f')
        model = VGG16()
        y = model(x)
        self.assertTrue(y.dtype == np.float32)

    def test_backward1(self):
        x = np.random.randn(2, 3, 224, 224).astype('f')
        _model = chainer.links.VGG16Layers(None)

        with chainer.using_config('train', False):
            out_layer_name = 'fc8'
            _y = _model.forward(x, [out_layer_name])[out_layer_name]
            _y.grad = np.ones_like(_y.data)
            _y.backward()

        model = VGG16()
        layers = _model.available_layers
        for l in layers:
            if "conv" in l or "fc" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                m1.W.data = m2.W.data
                m1.b.data = m2.b.data
                if "fc" in l:
                    m1.W.data = m1.W.data.T

        with test_mode():
            y = model(x)
            y.backward()

        layers = _model.available_layers
        for l in layers:
            if "conv" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                self.assertTrue(array_allclose(m1.W.data, m2.W.data))
                self.assertTrue(array_allclose(m1.b.data, m2.b.data))
            elif "fc" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                self.assertTrue(array_allclose(m1.W.data, m2.W.data.T))
                self.assertTrue(array_allclose(m1.b.data, m2.b.data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_sigmoid.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestSigmoid(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = CF.sigmoid(x)
        y = sigmoid(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        x = np.random.randn(10, 10).astype(np.float32)
        y2 = CF.sigmoid(x)
        y = sigmoid(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        self.assertTrue(gradient_check(sigmoid, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        self.assertTrue(gradient_check(sigmoid, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        self.assertTrue(gradient_check(sigmoid, x_data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_softmax.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestSoftmaxSimple(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = CF.softmax(x, axis=1)
        y = softmax_simple(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        np.random.seed(0)
        x = np.random.rand(10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = softmax_simple(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward3(self):
        np.random.seed(0)
        x = np.random.rand(10, 10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = softmax_simple(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: softmax_simple(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: softmax_simple(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: softmax_simple(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))


class TestSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = CF.softmax(x, axis=1)
        y = softmax(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        np.random.seed(0)
        x = np.random.rand(10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = softmax(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward3(self):
        np.random.seed(0)
        x = np.random.rand(10, 10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = softmax(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))


class TestSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = CF.softmax(x, axis=1)
        y = softmax(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        np.random.seed(0)
        x = np.random.rand(10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = softmax(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward3(self):
        np.random.seed(0)
        x = np.random.rand(10, 10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = softmax(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))


class TestLogSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        y = log_softmax(x)
        y2 = CF.log_softmax(x)
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]])
        f = lambda x: log_softmax(x)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(10, 10)
        f = lambda x: log_softmax(x)
        self.assertTrue(gradient_check(f, x))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_softmax_cross_entropy.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestSoftmaxCrossEntropy(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        y = softmax_cross_entropy(x, t)
        y2 = CF.softmax_cross_entropy(x, t)
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        f = lambda x: softmax_cross_entropy(x, Variable(t))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, CLS_NUM = 10, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        N, CLS_NUM = 100, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))


class TestSoftmaxCrossEntropy_simple(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        y = softmax_cross_entropy_simple(x, t)
        y2 = CF.softmax_cross_entropy(x, t)
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        f = lambda x: softmax_cross_entropy_simple(x, Variable(t))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, CLS_NUM = 10, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: softmax_cross_entropy_simple(x, t)
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        N, CLS_NUM = 100, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: softmax_cross_entropy_simple(x, t)
        self.assertTrue(gradient_check(f, x))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_sum.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestSum(unittest.TestCase):

    def test_datatype(self):
        x = Variable(np.random.rand(10))
        y = F_sum(x)
        # np.float64ではなく0次元のnp.ndarrayを返す
        self.assertFalse(np.isscalar(y))

    def test_forward1(self):
        x = Variable(np.array(2.0))
        y = F_sum(x)
        expected = np.sum(x.data)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward2(self):
        x = Variable(np.random.rand(10, 20, 30))
        y = F_sum(x, axis=1)
        expected = np.sum(x.data, axis=1)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward3(self):
        x = Variable(np.random.rand(10, 20, 30))
        y = F_sum(x, axis=1, keepdims=True)
        expected = np.sum(x.data, axis=1, keepdims=True)
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x_data = np.random.rand(10)
        f = lambda x: F_sum(x)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10)
        f = lambda x: F_sum(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20)
        f = lambda x: F_sum(x, axis=2)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.rand(10, 20, 20)
        f = lambda x: F_sum(x, axis=None)
        self.assertTrue(gradient_check(f, x_data))


class TestSumTo(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.rand(10))
        y = F_sum_to(x, (1,))
        expected = np.sum(x.data)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward2(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F_sum_to(x, (1, 3))
        expected = np.sum(x.data, axis=0, keepdims=True)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward3(self):
        x = Variable(np.random.rand(10))
        y = F_sum_to(x, (10,))
        expected = x.data  # 同じ形状なので何もしない
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x_data = np.random.rand(10)
        f = lambda x: F_sum_to(x, (1,))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10) * 10
        f = lambda x: F_sum_to(x, (10,))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20) * 100
        f = lambda x: F_sum_to(x, (10,))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.rand(10)
        f = lambda x: F_sum_to(x, (10,)) + 1
        self.assertTrue(gradient_check(f, x_data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_transpose.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestTranspose(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = transpose(x)
        self.assertEqual(y.shape, (3, 2))

    def test_backward1(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(gradient_check(transpose, x))

    def test_backward2(self):
        x = np.array([1, 2, 3])
        self.assertTrue(gradient_check(transpose, x))

    def test_backward3(self):
        x = np.random.randn(10, 5)
        self.assertTrue(gradient_check(transpose, x))

    def test_backward4(self):
        x = np.array([1, 2])
        self.assertTrue(gradient_check(transpose, x))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_vgg16.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestVGG16(unittest.TestCase):

    def test_forward1(self):
        x = np.random.randn(1, 3, 224, 224).astype('f')
        _model = chainer.links.VGG16Layers(None)

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                out_layer_name = 'fc8'
                _y = _model.forward(x, [out_layer_name])[out_layer_name]

        model = VGG16()
        layers = _model.available_layers
        for l in layers:
            if "conv" in l or "fc" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                m1.W.data = m2.W.data
                m1.b.data = m2.b.data
                if "fc" in l:
                    m1.W.data = m1.W.data.T

        with test_mode():
            y = model(x)

        self.assertTrue(array_allclose(y.data, _y.data))


    def test_forward2(self):
        x = np.random.randn(1, 3, 224, 224).astype('f')
        model = VGG16()
        y = model(x)
        self.assertTrue(y.dtype == np.float32)

    def test_backward1(self):
        x = np.random.randn(2, 3, 224, 224).astype('f')
        _model = chainer.links.VGG16Layers(None)

        with chainer.using_config('train', False):
            out_layer_name = 'fc8'
            _y = _model.forward(x, [out_layer_name])[out_layer_name]
            _y.grad = np.ones_like(_y.data)
            _y.backward()

        model = VGG16()
        layers = _model.available_layers
        for l in layers:
            if "conv" in l or "fc" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                m1.W.data = m2.W.data
                m1.b.data = m2.b.data
                if "fc" in l:
                    m1.W.data = m1.W.data.T

        with test_mode():
            y = model(x)
            y.backward()

        layers = _model.available_layers
        for l in layers:
            if "conv" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                self.assertTrue(array_allclose(m1.W.data, m2.W.data))
                self.assertTrue(array_allclose(m1.b.data, m2.b.data))
            elif "fc" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                self.assertTrue(array_allclose(m1.W.data, m2.W.data.T))
                self.assertTrue(array_allclose(m1.b.data, m2.b.data))


# =============================================================================
# =============================================================================
# =============================================================================
#
# test_weight_decay.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestWeightDecay(unittest.TestCase):

    def test_compare1(self):
        rate = 0.4
        x = np.random.rand(10, 2)
        t = np.zeros((10)).astype(int)
        layer = L_Linear(in_size=2, out_size=3, nobias=True)
        layer.W.data = np.ones_like(layer.W.data)
        optimizer = SGD().setup(layer)
        optimizer.add_hook(WeightDecay(rate=rate))

        layer.cleargrads()
        y = layer(x)
        y = softmax_cross_entropy(y, t)
        y.backward()
        optimizer.update()
        W0 = layer.W.data.copy()

        layer.W.data = np.ones_like(layer.W.data)
        optimizer.hooks.clear()
        layer.cleargrads()
        y = layer(x)
        y = softmax_cross_entropy(y, t) + rate / 2 * (layer.W ** 2).sum()
        y.backward()
        optimizer.update()
        W1 = layer.W.data
        self.assertTrue(array_allclose(W0, W1))


# =============================================================================
# =============================================================================
# =============================================================================
#
# 
#
# =============================================================================
# =============================================================================
# =============================================================================


