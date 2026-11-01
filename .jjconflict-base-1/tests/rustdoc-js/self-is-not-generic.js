// exact-check

const EXPECTED = [
    {
        'query': 'A -> A',
        'others': [
            { 'path': 'self_is_not_generic::Thing', 'name': 'from' }
        ],
    },
    {
        'query': 'A -> B',
        'others': [
            { 'path': 'self_is_not_generic::Thing', 'name': 'try_from' }
        ],
    },
    {
        'query': 'Combine -> Combine',
        'others': [
            { 'path': 'self_is_not_generic::Combine', 'name': 'combine' }
        ],
    }
];
