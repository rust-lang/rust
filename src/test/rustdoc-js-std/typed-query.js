// exact-check

const QUERY = 'macro:print';

const EXPECTED = {
    'others': [
        { 'path': 'std', 'name': 'print' },
        { 'path': 'std', 'name': 'eprint' },
        { 'path': 'std', 'name': 'println' },
        { 'path': 'std', 'name': 'eprintln' },
        { 'path': 'std::pin', 'name': 'pin' },
        { 'path': 'core::pin', 'name': 'pin' },
    ],
};
