// exact-check

const QUERY = [
    "i32",
    "str",
    "primitive:str",
    "struct:str",
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
        'returned': [
            { 'path': 'primitive', 'name': 'foo' },
        ],
    },
    {
        'returned': [],
    },
    {
        'others': [],
        'in_args': [],
        'returned': [],
    },
];
