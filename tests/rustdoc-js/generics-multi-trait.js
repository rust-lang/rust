// exact-check

const EXPECTED = [
    // check one of the generic items
    {
        'query': 'Result<SomeTrait>',
        'in_args': [
            { 'path': 'generics_multi_trait', 'name': 'beta' },
        ],
        'returned': [
            { 'path': 'generics_multi_trait', 'name': 'bet' },
        ],
    },
    {
        'query': 'Zzzzzzzzzzzzzzzzzz',
        'in_args': [
            { 'path': 'generics_multi_trait', 'name': 'beta' },
        ],
        'returned': [
            { 'path': 'generics_multi_trait', 'name': 'bet' },
        ],
    },
    // ignore the name of the generic itself
    {
        'query': 'Nonononononononono',
        'in_args': [],
        'returned': [],
    },
];
