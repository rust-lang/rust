const QUERY = '"error"';

const EXPECTED = {
    'others': [
        { 'path': 'std', 'name': 'error' },
        { 'path': 'std::fmt', 'name': 'Error' },
        { 'path': 'std::io', 'name': 'Error' },
    ],
    'in_args': [],
    'returned': [
        { 'path': 'std::fmt::LowerExp', 'name': 'fmt' },
    ],
};
