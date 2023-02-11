const QUERY = [
    'Result<SomeTrait>',
    'OtherThingxxxxxxxx',
];

const EXPECTED = [
    {
        'in_args': [
            { 'path': 'generics_trait', 'name': 'beta' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'bet' },
        ],
    },
    {
        'in_args': [
            { 'path': 'generics_trait', 'name': 'alpha' },
        ],
        'returned': [
            { 'path': 'generics_trait', 'name': 'alef' },
        ],
    },
];
