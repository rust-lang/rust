// exact-check

const QUERY = [
    'Result<SomeTrait>',
    'Zzzzzzzzzzzzzzzzzz',
    'Nonononononononono',
];

const EXPECTED = [
    // check one of the generic items
    {
        'in_args': [
            { 'path': 'generics_multi_trait', 'name': 'beta' },
        ],
        'returned': [
            { 'path': 'generics_multi_trait', 'name': 'bet' },
        ],
    },
    {
        'in_args': [
            { 'path': 'generics_multi_trait', 'name': 'beta' },
        ],
        'returned': [
            { 'path': 'generics_multi_trait', 'name': 'bet' },
        ],
    },
    // ignore the name of the generic itself
    {
        'in_args': [],
        'returned': [],
    },
];
