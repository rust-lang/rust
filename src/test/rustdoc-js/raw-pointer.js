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
            { 'path': 'raw_pointer::Ccccccc', 'name': 'eeeeeee' },
        ],
    },
    {
        // Aaaaaaa -> Aaaaaaa
        'others': [
            { 'path': 'raw_pointer::Ccccccc', 'name': 'fffffff' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ggggggg' },
        ],
    },
    {
        // Aaaaaaa -> usize
        'others': [],
    },
    {
        // -> Aaaaaaa
        'others': [
            { 'path': 'raw_pointer::Ccccccc', 'name': 'fffffff' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ggggggg' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ddddddd' },
            { 'path': 'raw_pointer', 'name': 'bbbbbbb' },
        ],
    },
    {
        // Aaaaaaa
        'others': [
            { 'path': 'raw_pointer', 'name': 'Aaaaaaa' },
        ],
        'in_args': [
            { 'path': 'raw_pointer::Ccccccc', 'name': 'fffffff' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ggggggg' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'eeeeeee' },
        ],
        'returned': [
            { 'path': 'raw_pointer::Ccccccc', 'name': 'fffffff' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ggggggg' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ddddddd' },
            { 'path': 'raw_pointer', 'name': 'bbbbbbb' },
        ],
    },
];
