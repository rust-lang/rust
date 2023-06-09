// exact-check

const QUERY = 'prinltn';
const FILTER_CRATE = 'std';

const EXPECTED = {
    'others': [
        { 'path': 'std', 'name': 'println' },
        { 'path': 'std', 'name': 'print' },
        { 'path': 'std', 'name': 'eprintln' },
    ],
};
