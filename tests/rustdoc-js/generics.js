// exact-check

const QUERY = [
    'R<P>',
    '"P"',
    'P',
    'ExtraCreditStructMulti<ExtraCreditInnerMulti, ExtraCreditInnerMulti>',
    'TraitCat',
    'TraitDog',
    'Result<String>',
];

const EXPECTED = [
    {
        // R<P>
        'returned': [
            { 'path': 'generics', 'name': 'alef' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
        ],
    },
    {
        // "P"
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
        // P
        'returned': [
            { 'path': 'generics', 'name': 'alef' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
        ],
    },
    {
        // "ExtraCreditStructMulti"<ExtraCreditInnerMulti, ExtraCreditInnerMulti>
        'in_args': [
            { 'path': 'generics', 'name': 'extracreditlabhomework' },
        ],
        'returned': [],
    },
    {
        // TraitCat
        'in_args': [
            { 'path': 'generics', 'name': 'gamma' },
        ],
    },
    {
        // TraitDog
        'in_args': [
            { 'path': 'generics', 'name': 'gamma' },
        ],
    },
    {
        // Result<String>
        'others': [],
        'returned': [
            { 'path': 'generics', 'name': 'super_soup' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'super_soup' },
        ],
    },
];
