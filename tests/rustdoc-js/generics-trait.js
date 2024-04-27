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
        'query': 'Resulx<SomeTrait>',
        'in_args': [],
        'returned': [],
    },
    {
        'query': 'Result<SomeTraiz>',
        'proposeCorrectionFrom': 'SomeTraiz',
        'proposeCorrectionTo': 'SomeTrait',
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
