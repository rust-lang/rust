// exact-check

const QUERY = [
    "i32",
    "str",
    "TotoIsSomewhere",
];

const EXPECTED = [
    {
        'in_args': [
            { 'path': 'primitive', 'name': 'foo' },
        ],
    },
    {
        'returned': [
            { 'path': 'primitive', 'name': 'foo' },
        ],
    },
    {
        'others': [],
        'in_args': [],
        'returned': [],
    },
];
