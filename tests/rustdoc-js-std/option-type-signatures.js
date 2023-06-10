const EXPECTED = [
    {
        'query': 'option, fnonce -> option',
        'others': [
            { 'path': 'std::option::Option', 'name': 'map' },
        ],
    },
    {
        'query': 'option -> default',
        'others': [
            { 'path': 'std::option::Option', 'name': 'unwrap_or_default' },
            { 'path': 'std::option::Option', 'name': 'get_or_insert_default' },
        ],
    },
];
