// ignore-order

const EXPECTED = [
    {
        'query': 'bufread -> result<[u8]>',
        'others': [
            { 'path': 'std::boxed::Box', 'name': 'fill_buf' },
        ],
    },
    {
        'query': 'split<bufread> -> option<result<vec<u8>>>',
        'others': [
            { 'path': 'std::io::Split', 'name': 'next' },
        ],
    },
];
