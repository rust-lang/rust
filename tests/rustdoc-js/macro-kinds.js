// exact-check

const EXPECTED = [
    {
        'query': 'macro:macro',
        'others': [
            { 'path': 'macro_kinds', 'name': 'macro1' },
            { 'path': 'macro_kinds', 'name': 'macro3' },
        ],
    },
    {
        'query': 'attr:macro',
        'others': [
            { 'path': 'macro_kinds', 'name': 'macro1' },
            { 'path': 'macro_kinds', 'name': 'macro2' },
        ],
    },
];
