const EXPECTED = [
    {
        'query': 'trait<nested>',
        'in_args': [
           { 'path': 'where_clause', 'name': 'abracadabra' },
        ],
    },
    {
        'query': '-> trait<nested>',
        'others': [
            { 'path': 'where_clause', 'name': 'alacazam' },
        ],
    },
    {
        'query': 't1, t2',
        'others': [
            { 'path': 'where_clause', 'name': 'presto' },
        ],
    },
    {
        'query': '-> shazam',
        'others': [
            { 'path': 'where_clause', 'name': 'bippety' },
            { 'path': 'where_clause::Drizzel', 'name': 'boppety' },
        ],
    },
    {
        'query': 'drizzel -> shazam',
        'others': [
            { 'path': 'where_clause::Drizzel', 'name': 'boppety' },
        ],
    },
];
