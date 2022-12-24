import unittest

#import cupy as np  # !! CUPY !!
import cupy as cp
import chainer
import chainer.functions as CF

from dezero.all import *


# =============================================================================
# =============================================================================
# =============================================================================
#
# gpu_test_batchnorm.py
#
# =============================================================================
# =============================================================================
# =============================================================================
def get_params(N, C, H=None, W=None, dtype='f'):
    if H is not None:
        x = cp.random.randn(N, C, H, W).astype(dtype)
    else:
        x = cp.random.randn(N, C).astype(dtype)
    gamma = cp.random.randn(C).astype(dtype)
    beta = cp.random.randn(C).astype(dtype)
    mean = cp.random.randn(C).astype(dtype)
    var = cp.abs(cp.random.randn(C).astype(dtype))
    return x, gamma, beta, mean, var


class TestFixedBatchNorm(unittest.TestCase):

    def test_type1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        with test_mode():
            y = batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(y.data.dtype == cp.float32)

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
        self.assertTrue(y.data.dtype == cp.float32)

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

    def test_backward1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=cp.float64)
        f = lambda x: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=cp.float64)
        f = lambda gamma: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, gamma))

    def test_backward3(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=cp.float64)
        f = lambda beta: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, beta))

    def test_backward4(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=cp.float64)
        f = lambda x: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, x))

    def test_backward5(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=cp.float64)
        f = lambda gamma: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, gamma))

    def test_backward6(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=cp.float64)
        f = lambda beta: batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, beta))


class TestBatchNormLayer(unittest.TestCase):

    def test_forward1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        ly = chainer.links.BatchNormalization(3)
        l = L_BatchNorm()
        ly.to_gpu()
        l.to_gpu()
        cy = ly(x)
        y = l(x)

        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward2(self):
        N, C = 8, 3
        cl = chainer.links.BatchNormalization(C)
        l = L_BatchNorm()
        cl.to_gpu()
        l.to_gpu()

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
# gpu_test_linear.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestLinear(unittest.TestCase):

    def test_forward1(self):
        x = Variable(cp.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        b = None
        y = linear(x, w, b)

        res = y.data
        expected = cp.array([[14, 32], [32, 77]])
        self.assertTrue(array_allclose(res, expected))

    def test_forward2(self):
        x = cp.array([[1, 2, 3], [4, 5, 6]]).astype('f')
        W = x.T
        b = None
        y = linear(x, W, b)

        cy = chainer.functions.linear(x, W.T)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward3(self):
        layer = chainer.links.Linear(3, 2)
        layer.to_gpu()
        x = cp.array([[1, 2, 3], [4, 5, 6]]).astype('f')
        W = layer.W.data.T
        b = layer.b.data
        y = linear(x, W, b)

        cy = layer(x)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_backward1(self):
        x = cp.random.randn(3, 2)
        W = cp.random.randn(2, 3)
        b = cp.random.randn(3)
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

    def test_backward1(self):
        x = cp.random.randn(3, 2)
        W = cp.random.randn(2, 3)
        b = cp.random.randn(3)
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = cp.random.randn(100, 200)
        W = cp.random.randn(200, 300)
        b = None
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))


# =============================================================================
# =============================================================================
# =============================================================================
#
# gpu_test_vgg16.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class TestVGG16(unittest.TestCase):

    def test_forward1(self):
        x = cp.random.randn(1, 3, 224, 224).astype('f')
        _model = chainer.links.VGG16Layers(None)
        _model.to_gpu()

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
        model.to_gpu()

        with test_mode():
            y = model(x)

        self.assertTrue(array_allclose(y.data, _y.data))

    def test_forward2(self):
        x = cp.random.randn(1, 3, 224, 224).astype('f')
        model = VGG16()
        model.to_gpu()
        y = model(x)
        self.assertTrue(y.dtype == cp.float32)

    def test_backward1(self):
        x = cp.random.randn(2, 3, 224, 224).astype('f')
        _model = chainer.links.VGG16Layers(None)
        _model.to_gpu()

        with chainer.using_config('train', False):
            out_layer_name = 'fc8'
            _y = _model.forward(x, [out_layer_name])[out_layer_name]
            _y.grad = cp.ones_like(_y.data)
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
        model.to_gpu()

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
# 
#
# =============================================================================
# =============================================================================
# =============================================================================


