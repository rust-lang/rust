// exact-check

const EXPECTED = [
    {
        'query': '()',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'side_effect' },
            { 'path': 'tuple_unit', 'name': 'one' },
            { 'path': 'tuple_unit', 'name': 'two' },
            { 'path': 'tuple_unit', 'name': 'nest' },
        ],
        'in_args': [],
    },
    {
        'query': 'primitive:unit',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'side_effect' },
        ],
        'in_args': [],
    },
    {
        'query': 'primitive:tuple',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'one' },
            { 'path': 'tuple_unit', 'name': 'two' },
            { 'path': 'tuple_unit', 'name': 'nest' },
        ],
        'in_args': [],
    },
    {
        'query': '(P)',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'not_tuple' },
            { 'path': 'tuple_unit', 'name': 'one' },
            { 'path': 'tuple_unit', 'name': 'two' },
        ],
        'in_args': [],
    },
    {
        'query': '(P,)',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'one' },
            { 'path': 'tuple_unit', 'name': 'two' },
        ],
        'in_args': [],
    },
    {
        'query': '(P, P)',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'two' },
        ],
        'in_args': [],
    },
    {
        'query': '(P, ())',
        'returned': [],
        'in_args': [],
    },
    {
        'query': '(Q, R<()>)',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'nest' },
        ],
        'in_args': [],
    },
    {
        'query': '(R)',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'nest' },
        ],
        'in_args': [],
    },
    {
        'query': 'R<(u32)>',
        'returned': [
            { 'path': 'tuple_unit', 'name': 'nest' },
        ],
        'in_args': [],
    },
];
