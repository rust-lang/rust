// exact-check

const EXPECTED = [
    {
        'query': 'i32',
        'in_args': [
            { 'path': 'primitive', 'name': 'foo' },
        ],
    },
    {
        'query': 'str',
        'returned': [
            { 'path': 'primitive', 'name': 'foo' },
        ],
    },
    {
        'query': 'primitive:str',
        'returned': [
            { 'path': 'primitive', 'name': 'foo' },
        ],
    },
    {
        'query': 'struct:str',
        'returned': [],
    },
    {
        'query': 'TotoIsSomewhere',
        'others': [],
        'in_args': [],
        'returned': [],
    },
];
