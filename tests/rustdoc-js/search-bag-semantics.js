// exact-check

const QUERY = [
    'P',
    'P, P',
];

const EXPECTED = [
    {
        'in_args': [
            { 'path': 'search_bag_semantics', 'name': 'alacazam' },
            { 'path': 'search_bag_semantics', 'name': 'abracadabra' },
        ],
    },
    {
        'others': [
            { 'path': 'search_bag_semantics', 'name': 'abracadabra' },
        ],
    },
];
