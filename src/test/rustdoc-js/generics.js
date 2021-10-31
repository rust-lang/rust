// exact-check

const QUERY = [
    '"R<P>"',
    '"P"',
    'P',
    '"ExtraCreditStructMulti<ExtraCreditInnerMulti, ExtraCreditInnerMulti>"',
    'TraitCat',
    'TraitDog',
];

const EXPECTED = [
    {
        'returned': [
            { 'path': 'generics', 'name': 'alef' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
        ],
    },
    {
        'others': [
            { 'path': 'generics', 'name': 'P' },
        ],
        'returned': [
            { 'path': 'generics', 'name': 'alef' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
        ],
    },
    {
        'returned': [
            { 'path': 'generics', 'name': 'alef' },
            { 'path': 'generics', 'name': 'bet' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
            { 'path': 'generics', 'name': 'beta' },
        ],
    },
    {
        'in_args': [
            { 'path': 'generics', 'name': 'extracreditlabhomework' },
        ],
        'returned': [],
    },
    {
        'in_args': [
            { 'path': 'generics', 'name': 'gamma' },
        ],
    },
    {
        'in_args': [
            { 'path': 'generics', 'name': 'gamma' },
        ],
    },
];
