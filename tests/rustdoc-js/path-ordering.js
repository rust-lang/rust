// exact-check

const EXPECTED = {
    'query': 'bbbbbb::ccccccc',
    'others': [
        // `ccccccc` is an exact match for all three of these.
        // However `b` is a closer match for `bb` than for any
        // of the others, so it ought to go first.
        { 'path': 'path_ordering::bbbbbb', 'name': 'Ccccccc' },
        { 'path': 'path_ordering::abbbbb', 'name': 'Ccccccc' },
        { 'path': 'path_ordering::dbbbbb', 'name': 'Ccccccc' },
    ],
};
