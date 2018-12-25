// exact-check

const QUERY = 'waker_from';

const EXPECTED = {
    'others': [
        { 'path': 'std::task', 'name': 'local_waker_from_nonlocal' },
        { 'path': 'alloc::task', 'name': 'local_waker_from_nonlocal' },
    ],
};
