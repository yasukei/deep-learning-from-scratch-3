import contextlib
import gzip
import math
import os
import pickle
import subprocess
import tarfile
import urllib.request
import weakref


import numpy as np

# =============================================================================
# =============================================================================
# =============================================================================
#
# core.py
#
# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def test_mode():
    return using_config('train', False)

# =============================================================================
# Variable / Function
# =============================================================================
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, other):
        return pow(self, other)

    def __getitem__(self, other):
        return get_item(self, other)

    def matmul(self, other):
        return matmul(self, other)

    def dot(self, other):
        return matmul(self, other)

    def max(self, other):
        return F_max(self, other)

    def min(self, other):
        return F_min(self, other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return transpose(self, axes)

    @property
    def T(self):
        return transpose(self)

    def sum(self, axis=None, keepdims=False):
        return F_sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = as_cupy(self.data)


class Parameter(Variable):
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = F_sum_to(gx0, self.x0_shape)
            gx1 = F_sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1, get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = F_sum_to(gx0, x0.shape)
            gx1 = F_sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1, get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = F_sum_to(gx0, self.x0_shape)
            gx1 = F_sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1, get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = F_sum_to(gx0, x0.shape)
            gx1 = F_sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)



# =============================================================================
# =============================================================================
# =============================================================================
#
# cuda.py
#
# =============================================================================
# =============================================================================
# =============================================================================
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False


def get_array_module(x):
    """Returns the array module for `x`.

    Args:
        x (dezero.Variable or numpy.ndarray or cupy.ndarray): Values to
            determine whether NumPy or CuPy should be used.

    Returns:
        module: `cupy` or `numpy` is returned based on the argument.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    """Convert to `numpy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `numpy.ndarray`.
    Returns:
        `numpy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    """Convert to `cupy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `cupy.ndarray`.
    Returns:
        `cupy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    return cp.asarray(x)


# =============================================================================
# =============================================================================
# =============================================================================
#
# dataloaders.py
#
# =============================================================================
# =============================================================================
# =============================================================================
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                       range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t


# =============================================================================
# =============================================================================
# =============================================================================
#
# transforms.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class Compose:
    """Compose several transforms.

    Args:
        transforms (list): list of transforms
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# Transforms for NumPy ndarray
# =============================================================================
class Normalize:
    """Normalize a NumPy array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence):
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    """Flatten a NumPy array.
    """
    def __call__(self, array):
        return array.flatten()


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType



# =============================================================================
# =============================================================================
# =============================================================================
#
# datasets.py
#
# =============================================================================
# =============================================================================
# =============================================================================
class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]),\
                   self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


# =============================================================================
# Toy datasets
# =============================================================================
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


# =============================================================================
# MNIST-like dataset: MNIST / CIFAR /
# =============================================================================
class MNIST(Dataset):

    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(),
                                     Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        # plt.imshow(img, cmap='gray', interpolation='nearest')
        # plt.axis('off')
        # plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


class CIFAR10(Dataset):

    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return
        filepath = get_file(url)
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32))
            self.label = np.empty((50000), dtype=np.int)
            for i in range(5):
                self.data[i * 10000:(i + 1) * 10000] = self._load_data(
                    filepath, i + 1, 'train')
                self.label[i * 10000:(i + 1) * 10000] = self._load_label(
                    filepath, i + 1, 'train')
        else:
            self.data = self._load_data(filepath, 5, 'test')
            self.label = self._load_label(filepath, 5, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)


    def _load_data(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.array(data_dict[b'labels'])

    def show(self, row=10, col=10):
        H, W = 32, 32
        img = np.zeros((H*row, W*col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255
        # plt.imshow(img, interpolation='nearest')
        # plt.axis('off')
        # plt.show()

    @staticmethod
    def labels():
        return {0: 'ariplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


class CIFAR100(CIFAR10):

    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None,
                 label_type='fine'):
        assert label_type in ['fine', 'coarse']
        self.label_type = label_type
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return

        filepath = get_file(url)
        if self.train:
            self.data = self._load_data(filepath, 'train')
            self.label = self._load_label(filepath, 'train')
        else:
            self.data = self._load_data(filepath, 'test')
            self.label = self._load_label(filepath, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)

    def _load_data(self, filename, data_type='train'):
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filename, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if self.label_type == 'fine':
                        return np.array(data_dict[b'fine_labels'])
                    elif self.label_type == 'coarse':
                        return np.array(data_dict[b'coarse_labels'])

    @staticmethod
    def labels(label_type='fine'):
        coarse_labels = dict(enumerate(['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical device','household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees','vehicles 1','vehicles 2']))
        fine_labels = []
        return fine_labels if label_type == 'fine' else coarse_labels





# =============================================================================
# Big datasets
# =============================================================================
class ImageNet(Dataset):

    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels


# =============================================================================
# Sequential datasets: SinCurve, Shapekspare
# =============================================================================
class SinCurve(Dataset):

    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


class Shakespear(Dataset):

    def prepare(self):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        file_name = 'shakespear.txt'
        path = get_file(url, file_name)
        with open(path, 'r') as f:
            data = f.read()
        chars = list(data)

        char_to_id = {}
        id_to_char = {}
        for word in data:
            if word not in char_to_id:
                new_id = len(char_to_id)
                char_to_id[word] = new_id
                id_to_char[new_id] = word

        indices = np.array([char_to_id[c] for c in chars])
        self.data = indices[:-1]
        self.label = indices[1:]
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


# =============================================================================
# Utils
# =============================================================================
def load_cache_npz(filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)
    if not os.path.exists(filepath):
        return None, None

    loaded = np.load(filepath)
    return loaded['data'], loaded['label']

def save_cache_npz(data, label, filename, train=False):
    filename = filename[filename.rfind('/') + 1:]
    prefix = '.train.npz' if train else '.test.npz'
    filepath = os.path.join(cache_dir, filename + prefix)

    if os.path.exists(filepath):
        return

    print("Saving: " + filename + prefix)
    try:
        np.savez_compressed(filepath, data=data, label=label)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    print(" Done")
    return filepath


# =============================================================================
# =============================================================================
# =============================================================================
#
# functions_conv.py
#
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# [simple version] conv2d_simple / pooling_simple
# =============================================================================
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


# =============================================================================
#  conv2d / deconv2d
# =============================================================================
class F_Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        xp = get_array_module(x)

        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # ==== gx ====
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return F_Conv2d(stride, pad)(x, W, b)


class F_Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        xp = get_array_module(x)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = xp.tensordot(Weight, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return F_Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# =============================================================================
#  pooling(max-pooling) / average_pooling
# =============================================================================
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))
        
        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        # TODO(Koki): This is simple implementation
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= (KW*KH)
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N*C*OH*OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(gcol, self.input_shape, self.kernel_size, self.stride,
                    self.pad, to_matrix=False)
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)


# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    """Extract patches from an image based on the filter.

    Args:
        x (`dezero.Variable` or `ndarray`): Input variable of shape
            `(N, C, H, W)`
        kernel_size (int or (int, int)): Size of kernel.
        stride (int or (int, int)): Stride of kernel.
        pad (int or (int, int)): Spatial padding width for input arrays.
        to_matrix (bool): If True the `col` will be reshaped to 2d array whose
            shape is `(N*OH*OW, C*KH*KW)`

    Returns:
        `dezero.Variable`: Output variable. If the `to_matrix` is False, the
            output shape is `(N, C, KH, KW, OH, OW)`, otherwise
            `(N*OH*OW, C*KH*KW)`.

    Notation:
    - `N` is the batch size.
    - `C` is the number of the input channels.
    - `H` and `W` are the height and width of the input image, respectively.
    - `KH` and `KW` are the height and width of the filters, respectively.
    - `SH` and `SW` are the strides of the filter.
    - `PH` and `PW` are the spatial padding sizes.
    - `OH` and `OW` are the the height and width of the output, respectively.
    """
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
#  numpy im2col
# =============================================================================
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img,
                     ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                     mode='constant', constant_values=(0,))
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH:H + PH, PW:W + PW]


def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cupy.empty((n, c, h, w), dtype=col.dtype)

    cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img

# =============================================================================
# =============================================================================
# =============================================================================
#
# functions.py
#
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


def expand_dims(x, axis):
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def F_sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = U_sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def F_sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = F_sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)


mean = average


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class F_Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else F_sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return F_Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # Release t.data (ndarray) for memory efficiency
    return y


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    def forward(self, x):
        xp = get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


# =============================================================================
# loss function: mean_squared_error / softmax_cross_entropy / sigmoid_cross_entropy / binary_cross_entropy
# =============================================================================
def mean_squared_error_simple(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    y = sum(diff ** 2) / len(diff)
    return y


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x, t):
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p, t):
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================
def accuracy(y, t):
    """
    [WAR] This function is not differentiable.
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if Config.train:
        xp = get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = get_array_module(x)

        if Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1. if m - 1. > 1. else 1.
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_nrom(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


def embed_id(x, W):
    return W[x]


# =============================================================================
# max / min / clip
# =============================================================================
class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def F_max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def F_min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


# =============================================================================
# =============================================================================
# =============================================================================
#
# layers.py
#
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Layer (base class)
# =============================================================================
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================
class L_Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        y = linear(x, self.W, self.b)
        return y


class L_Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class L_Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        """An Elman RNN with tanh.

        Args:
            hidden_size (int): The number of features in the hidden state.
            in_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.

        """
        super().__init__()
        self.x2h = L_Linear(hidden_size, in_size=in_size)
        self.h2h = L_Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        self.x2f = L_Linear(H, in_size=I)
        self.x2i = L_Linear(H, in_size=I)
        self.x2o = L_Linear(H, in_size=I)
        self.x2u = L_Linear(H, in_size=I)
        self.h2f = L_Linear(H, in_size=H, nobias=True)
        self.h2i = L_Linear(H, in_size=H, nobias=True)
        self.h2o = L_Linear(H, in_size=H, nobias=True)
        self.h2u = L_Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new


# =============================================================================
# EmbedID / BatchNorm
# =============================================================================
class EmbedID(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_nrom(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)


# =============================================================================
# =============================================================================
# =============================================================================
#
# models.py
#
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L_Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


# =============================================================================
# VGG
# =============================================================================
class VGG16(Model):
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1_1 = L_Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L_Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L_Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L_Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L_Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L_Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L_Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L_Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L_Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L_Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L_Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L_Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L_Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L_Linear(4096)
        self.fc7 = L_Linear(4096)
        self.fc8 = L_Linear(1000)

        if pretrained:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:, :, ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image


# =============================================================================
# ResNet
# =============================================================================
class ResNet(Model):
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/resnet{}.npz'

    def __init__(self, n_layers=152, pretrained=False):
        super().__init__()

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        self.conv1 = L_Conv2d(64, 7, 2, 3)
        self.bn1 = L.BatchNorm()
        self.res2 = BuildingBlock(block[0], 64, 64, 256, 1)
        self.res3 = BuildingBlock(block[1], 256, 128, 512, 2)
        self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2)
        self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2)
        self.fc6 = L_Linear(1000)

        if pretrained:
            weights_path = utils.get_file(ResNet.WEIGHTS_PATH.format(n_layers))
            self.load_weights(weights_path)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.pooling(x, kernel_size=3, stride=2)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = _global_average_pooling_2d(x)
        x = self.fc6(x)
        return x


class ResNet152(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(152, pretrained)


class ResNet101(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(101, pretrained)


class ResNet50(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(50, pretrained)


def _global_average_pooling_2d(x):
    N, C, H, W = x.shape
    h = F.average_pooling(x, (H, W), stride=1)
    h = F.reshape(h, (N, C))
    return h


class BuildingBlock(Layer):
    def __init__(self, n_layers=None, in_channels=None, mid_channels=None,
                 out_channels=None, stride=None, downsample_fb=None):
        super().__init__()

        self.a = BottleneckA(in_channels, mid_channels, out_channels, stride,
                             downsample_fb)
        self._forward = ['a']
        for i in range(n_layers - 1):
            name = 'b{}'.format(i+1)
            bottleneck = BottleneckB(out_channels, mid_channels)
            setattr(self, name, bottleneck)
            self._forward.append(name)

    def forward(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x


class BottleneckA(Layer):
    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, downsample_fb=False):
        super().__init__()
        # In the original MSRA ResNet, stride=2 is on 1x1 convolution.
        # In Facebook ResNet, stride=2 is on 3x3 convolution.
        stride_1x1, stride_3x3 = (1, stride) if downsample_fb else (stride, 1)
       
        self.conv1 = L_Conv2d(mid_channels, 1, stride_1x1, 0,
                              nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L_Conv2d(mid_channels, 3, stride_3x3, 1,
                              nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L_Conv2d(out_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()
        self.conv4 = L_Conv2d(out_channels, 1, stride, 0,
                              nobias=True)
        self.bn4 = L.BatchNorm()

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(Layer):
    """A bottleneck layer that maintains the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        
        self.conv1 = L_Conv2d(mid_channels, 1, 1, 0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L_Conv2d(mid_channels, 3, 1, 1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L_Conv2d(in_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


# =============================================================================
# SqueezeNet
# =============================================================================
class SqueezeNet(Model):
    def __init__(self, pretrained=False):
        pass

    def forward(self, x):
        pass

# =============================================================================
# =============================================================================
# =============================================================================
#
# optimizers.py
#
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Optimizer (base class)
# =============================================================================
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


# =============================================================================
# Hook functions
# =============================================================================
class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data


class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate


class FreezeParam:
    def __init__(self, *layers):
        self.freeze_params = []
        for l in layers:
            if isinstance(l, Parameter):
                self.freeze_params.append(l)
            else:
                for p in l.params():
                    self.freeze_params.append(p)

    def __call__(self, params):
        for p in self.freeze_params:
            p.grad = None



# =============================================================================
# SGD / MomentumSGD / AdaGrad / AdaDelta / Adam
# =============================================================================
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (xp.sqrt(h) + eps)


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        xp = get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)


# =============================================================================
# =============================================================================
# =============================================================================
#
# utils.py
#
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Visualize for computational graph
# =============================================================================
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f):
    # for function
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # for edge
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += dot_edge.format(id(f), id(y()))
    return ret


def get_dot_graph(output, verbose=True):
    """Generates a graphviz DOT text of a computational graph.

    Build a graph of functions and variables backward-reachable from the
    output. To visualize a graphviz DOT text, you need the dot binary from the
    graphviz package (www.graphviz.org).

    Args:
        output (dezero.Variable): Output variable from which the graph is
            constructed.
        verbose (bool): If True the dot graph contains additional information
            such as shapes and dtypes.

    Returns:
        str: A graphviz DOT text consisting of nodes and edges that are
            backward-reachable from the output
    """
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]  # Extension(e.g. png, pdf)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass



# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================
def U_sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
    xp = get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


# =============================================================================
# Gradient check
# =============================================================================
def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwargs):
    """Test backward procedure of a given function.

    This automatically checks the backward-process of a given function. For
    checking the correctness, this function compares gradients by
    backprop and ones by numerical derivation. If the result is within a
    tolerance this function return True, otherwise False.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A traget `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        bool: Return True if the result is within a tolerance, otherwise False.
    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

    if not res:
        print('')
        print('========== FAILED (Gradient Check) ==========')
        print('Numerical Grad')
        print(' shape: {}'.format(num_grad.shape))
        val = str(num_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
        print('Backprop Grad')
        print(' shape: {}'.format(bp_grad.shape))
        val = str(bp_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
    return res


def numerical_grad(f, x, *args, **kwargs):
    """Computes numerical gradient by finite differences.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A target `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        `ndarray`: Gradient.
    """
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    xp = get_array_module(x)
    if xp is not np:
        np_x = as_numpy(x)
    else:
        np_x = x
    grad = xp.zeros_like(x)

    it = np.nditer(np_x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + eps
        y1 = f(x, *args, **kwargs)  # f(x+h)
        if isinstance(y1, Variable):
            y1 = y1.data
        y1 = y1.copy()

        x[idx] = tmp_val - eps
        y2 = f(x, *args, **kwargs)  # f(x-h)
        if isinstance(y2, Variable):
            y2 = y2.data
        y2 = y2.copy()

        diff = (y1 - y2).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    return grad


def array_equal(a, b):
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays
            to compare

    Returns:
        bool: True if the two arrays are equal.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = as_numpy(a), as_numpy(b)
    return np.array_equal(a, b)


def array_allclose(a, b, rtol=1e-4, atol=1e-5):
    """Returns True if two arrays(or variables) are element-wise equal within a
    tolerance.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays
            to compare
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.

    Returns:
        bool: True if the two arrays are equal within the given tolerance,
            False otherwise.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = as_numpy(a), as_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


# =============================================================================
# download function
# =============================================================================
def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# others
# =============================================================================
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

