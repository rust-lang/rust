// exact-check

// This test ensures that literal search is always applied on elements of the path.

const EXPECTED = [
    {
        'query': '"some::path"',
        'others': [
            { 'path': 'literal_path::some', 'name': 'Path' },
        ],
    },
    {
        'query': '"somea::path"',
        'others': [
            { 'path': 'literal_path::somea', 'name': 'Path' },
        ],
    },
    {
        'query': '"soma::path"',
        'others': [
        ],
    },
];
