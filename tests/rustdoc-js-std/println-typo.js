// exact-check

const FILTER_CRATE = 'std';

const EXPECTED = {
    'query': 'prinltn',
    'others': [
        { 'path': 'std', 'name': 'println' },
        { 'path': 'std', 'name': 'print' },
        { 'path': 'std', 'name': 'eprintln' },
    ],
};
