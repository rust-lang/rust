// exact-check

const FILTER_CRATE = 'std';

const EXPECTED = {
    'query': 'macro:print',
    'others': [
        { 'path': 'std', 'name': 'print' },
        { 'path': 'std', 'name': 'println' },
        { 'path': 'std', 'name': 'eprint' },
        { 'path': 'std', 'name': 'eprintln' },
    ],
};
