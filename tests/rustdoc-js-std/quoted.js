// ignore-order

const FILTER_CRATE = 'std';

const EXPECTED = {
    'query': '"error"',
    'others': [
        { 'path': 'std', 'name': 'error' },
        { 'path': 'std::fmt', 'name': 'Error' },
        { 'path': 'std::io', 'name': 'Error' },
    ],
    'in_args': [
        { 'path': 'std::fmt::Error', 'name': 'eq' },
        { 'path': 'std::fmt::Error', 'name': 'cmp' },
        { 'path': 'std::fmt::Error', 'name': 'partial_cmp' },

    ],
    'returned': [
        { 'path': 'std::fmt::LowerExp', 'name': 'fmt' },
    ],
};
