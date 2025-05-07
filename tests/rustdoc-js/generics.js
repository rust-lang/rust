// exact-check
// ignore-order

const EXPECTED = [
    {
        'query': 'R<P>',
        'returned': [
            { 'path': 'generics', 'name': 'alef' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
        ],
    },
    {
        'query': 'R<struct:P>',
        'returned': [
            { 'path': 'generics', 'name': 'alef' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
        ],
    },
    {
        'query': 'R<enum:P>',
        'returned': [],
        'in_args': [],
    },
    {
        'query': '"P"',
        'others': [
            { 'path': 'generics', 'name': 'P' },
        ],
        'returned': [],
        'in_args': [],
    },
    {
        'query': 'P',
        'returned': [],
        'in_args': [],
    },
    {
        'query': '"ExtraCreditStructMulti"<ExtraCreditInnerMulti, ExtraCreditInnerMulti>',
        'in_args': [
            { 'path': 'generics', 'name': 'extracreditlabhomework' },
        ],
        'returned': [],
    },
    {
        'query': 'TraitCat',
        'in_args': [
            { 'path': 'generics', 'name': 'gamma' },
        ],
    },
    {
        'query': 'TraitDog',
        'in_args': [
            { 'path': 'generics', 'name': 'gamma' },
        ],
    },
    {
        'query': 'Result<String>',
        'others': [],
        'returned': [
            { 'path': 'generics', 'name': 'super_soup' },
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'super_soup' },
        ],
    },
];
