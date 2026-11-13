// exact-check
// ignore-order

const EXPECTED = [
    {
        'query': 'Aaaaaaa -> i32',
        'others': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'eeeeeee' },
        ],
    },
    {
        'query': 'Aaaaaaa -> Aaaaaaa',
        'others': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
        ],
    },
    {
        'query': 'Aaaaaaa -> usize',
        'others': [],
    },
    {
        'query': '-> Aaaaaaa',
        'others': [
            { 'path': 'impl_trait', 'name': 'bbbbbbb' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'ddddddd' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'ggggggg' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
        ],
    },
    {
        'query': '-> Bbbbbbb',
        'others': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'ggggggg' },
        ],
    },
    {
        'query': 'Aaaaaaa',
        'others': [
            { 'path': 'impl_trait', 'name': 'Aaaaaaa' },
        ],
        'in_args': [
            { 'path': 'impl_trait::Ccccccc', 'name': 'eeeeeee' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
        ],
        'returned': [
            { 'path': 'impl_trait', 'name': 'bbbbbbb' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'ddddddd' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'ggggggg' },
            { 'path': 'impl_trait::Ccccccc', 'name': 'fffffff' },
        ],
    },
];
