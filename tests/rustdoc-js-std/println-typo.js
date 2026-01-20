// exact-check

const FILTER_CRATE = 'std';

const EXPECTED = {
    'query': 'prinltn',
    'others': [
        { 'path': 'std', 'name': 'println' },
        { 'path': 'std::prelude::v1', 'name': 'println' },
        { 'path': 'std', 'name': 'print' },
        { 'path': 'std::prelude::v1', 'name': 'print' },
        { 'path': 'std', 'name': 'eprintln' },
        { 'path': 'std::prelude::v1', 'name': 'eprintln' },
    ],
};
