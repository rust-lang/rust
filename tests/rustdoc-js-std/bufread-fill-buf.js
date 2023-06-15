// ignore-order

const EXPECTED = [
    {
        'query': 'bufread -> result<u8>',
        'others': [
            { 'path': 'std::io::Split', 'name': 'next' },
            { 'path': 'std::boxed::Box', 'name': 'fill_buf' },
            { 'path': 'std::io::Chain', 'name': 'fill_buf' },
            { 'path': 'std::io::Take', 'name': 'fill_buf' },
        ],
    },
];
