// ignore-order

const EXPECTED = [
    {
        'query': 'Aaaaaaa -> i32',
        'others': [
            { 'path': 'raw_pointer::Ccccccc', 'name': 'eeeeeee' },
        ],
    },
    {
        'query': 'Aaaaaaa -> Aaaaaaa',
        'others': [
            { 'path': 'raw_pointer::Ccccccc', 'name': 'fffffff' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ggggggg' },
        ],
    },
    {
        'query': 'Aaaaaaa -> usize',
        'others': [],
    },
    {
        'query': '-> Aaaaaaa',
        'others': [
            { 'path': 'raw_pointer::Ccccccc', 'name': 'fffffff' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ggggggg' },
            { 'path': 'raw_pointer::Ccccccc', 'name': 'ddddddd' },
            { 'path': 'raw_pointer', 'name': 'bbbbbbb' },
        ],
    },
    {
        'query': 'Aaaaaaa',
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
