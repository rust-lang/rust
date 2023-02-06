// exact-check

const QUERY = 'macro:print';
const FILTER_CRATE = 'std';

const EXPECTED = {
    'others': [
        { 'path': 'std', 'name': 'print' },
        { 'path': 'std', 'name': 'println' },
        { 'path': 'std', 'name': 'eprint' },
        { 'path': 'std', 'name': 'eprintln' },
    ],
};
