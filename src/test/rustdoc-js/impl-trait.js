// ignore-order

const QUERY = [
    'Aaaaaaa -> i32',
    'Aaaaaaa -> Aaaaaaa',
    'Aaaaaaa -> usize',
    '-> Aaaaaaa',
    'Aaaaaaa',
];

const EXPECTED = [
    {
        // Aaaaaaa -> i32
        'others': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'eeeeeee' },
        ],
    },
    {
        // Aaaaaaa -> Aaaaaaa
        'others': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
        ],
    },
    {
        // Aaaaaaa -> usize
        'others': [],
    },
    {
        // -> Aaaaaaa
        'others': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'ddddddd' },
            { 'path': 'impl_trait', 'name': 'bbbbbbb' },
        ],
    },
    {
        // Aaaaaaa
        'others': [
            { 'path': 'impl_trait', 'name': 'Aaaaaaa' },
        ],
        'in_args': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'eeeeeee' },
        ],
        'returned': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'ddddddd' },
            { 'path': 'impl_trait', 'name': 'bbbbbbb' },
        ],
    },
];
