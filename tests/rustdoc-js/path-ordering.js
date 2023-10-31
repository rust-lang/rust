// exact-check

const EXPECTED = {
    'query': 'b::ccccccc',
    'others': [
        // `ccccccc` is an exact match for all three of these.
        // However `b` is a closer match for `bb` than for any
        // of the others, so it ought to go first.
        { 'path': 'path_ordering::bb', 'name': 'Ccccccc' },
        { 'path': 'path_ordering::aa', 'name': 'Ccccccc' },
        { 'path': 'path_ordering::dd', 'name': 'Ccccccc' },
    ],
};
