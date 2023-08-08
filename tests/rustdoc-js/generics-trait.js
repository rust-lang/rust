// exact-check

const EXPECTED = [
    {
        'query': 'Result<SomeTrait>',
        'correction': null,
        'in_args': [
            { 'path': 'generics_trait', 'name': 'beta' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'bet' },
        ],
    },
    {
        'query': 'Result<SomeTraiz>',
        'correction': null,
        'in_args': [],
        'returned': [],
    },
    {
        'query': 'OtherThingxxxxxxxx',
        'correction': null,
        'in_args': [
            { 'path': 'generics_trait', 'name': 'alpha' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'alef' },
        ],
    },
    {
        'query': 'OtherThingxxxxxxxy',
        'correction': 'OtherThingxxxxxxxx',
        'in_args': [
            { 'path': 'generics_trait', 'name': 'alpha' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'alef' },
        ],
    },
];
