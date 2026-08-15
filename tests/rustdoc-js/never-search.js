// exact-check

const EXPECTED = [
    {
        'query': '! ->',
        'others': [
            { 'path': 'never_search', 'name': 'impossible' },
            { 'path': 'never_search', 'name': 'box_impossible' },
        ],
    },
    {
        'query': '-> !',
        'others': [
            { 'path': 'never_search', 'name': 'loops' },
        ],
    },
    {
        'query': '-> never',
        'others': [
            { 'path': 'never_search', 'name': 'loops' },
            { 'path': 'never_search', 'name': 'returns' },
        ],
    },
    {
        'query': '!',
        'in_args': [
            { 'path': 'never_search', 'name': 'impossible' },
            { 'path': 'never_search', 'name': 'box_impossible' },
        ],
    },
    {
        'query': 'never',
        'in_args': [
            { 'path': 'never_search', 'name': 'impossible' },
            { 'path': 'never_search', 'name': 'uninteresting' },
            { 'path': 'never_search', 'name': 'box_impossible' },
            { 'path': 'never_search', 'name': 'box_uninteresting' },
        ],
    },
    {
        'query': 'box<!>',
        'in_args': [
            { 'path': 'never_search', 'name': 'box_impossible' },
        ],
    },
    {
        'query': 'box<never>',
        'in_args': [
            { 'path': 'never_search', 'name': 'box_impossible' },
            { 'path': 'never_search', 'name': 'box_uninteresting' },
        ],
    },
    {
        'query': 'box<item=!>',
        'in_args': [],
        'returned': [],
    },
    {
        'query': 'box<item=never>',
        'in_args': [],
        'returned': [],
    },
];
