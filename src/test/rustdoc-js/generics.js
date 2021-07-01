// exact-check

const QUERY = [
  '"R<P>"',
  '"P"',
  'P',
  '"ExtraCreditStructMulti<ExtraCreditInnerMulti, ExtraCreditInnerMulti>"',
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
        ],
        'in_args': [
            { 'path': 'generics', 'name': 'alpha' },
        ],
    },
    {
        'in_args': [
            { 'path': 'generics', 'name': 'extracreditlabhomework' },
        ],
        'returned': [],
    },
];
