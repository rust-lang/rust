// exact-check

const QUERY = [
    'Result<SomeTrait>',
    'Result<SomeTraiz>',
    'OtherThingxxxxxxxx',
    'OtherThingxxxxxxxy',
];

const CORRECTIONS = [
    null,
    null,
    null,
    'OtherThingxxxxxxxx',
];

const EXPECTED = [
    // Result<SomeTrait>
    {
        'in_args': [
            { 'path': 'generics_trait', 'name': 'beta' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'bet' },
        ],
    },
    // Result<SomeTraiz>
    {
        'in_args': [],
        'returned': [],
    },
    // OtherThingxxxxxxxx
    {
        'in_args': [
            { 'path': 'generics_trait', 'name': 'alpha' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'alef' },
        ],
    },
    // OtherThingxxxxxxxy
    {
        'in_args': [
            { 'path': 'generics_trait', 'name': 'alpha' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'alef' },
        ],
    },
];
