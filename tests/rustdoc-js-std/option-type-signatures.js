const QUERY = [
    'option, fnonce -> option',
    'option -> default',
];

const EXPECTED = [
    {
        'others': [
            { 'path': 'std::option::Option', 'name': 'map' },
        ],
    },
    {
        'others': [
            { 'path': 'std::option::Option', 'name': 'unwrap_or_default' },
            { 'path': 'std::option::Option', 'name': 'get_or_insert_default' },
        ],
    },
];
