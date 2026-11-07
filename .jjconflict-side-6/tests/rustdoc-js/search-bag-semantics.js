// exact-check

const EXPECTED = [
    {
        'query': 'P',
        'in_args': [
            { 'path': 'search_bag_semantics', 'name': 'alacazam' },
            { 'path': 'search_bag_semantics', 'name': 'abracadabra' },
        ],
    },
    {
        'query': 'P, P',
        'others': [
            { 'path': 'search_bag_semantics', 'name': 'abracadabra' },
        ],
    },
];
