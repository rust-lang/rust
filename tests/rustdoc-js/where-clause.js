const QUERY = ['trait<nested>', '-> trait<nested>', 't1, t2', '-> shazam', 'drizzel -> shazam'];

const EXPECTED = [
    {
        'in_args': [
           { 'path': 'where_clause', 'name': 'abracadabra' },
        ],
    },
    {
        'others': [
            { 'path': 'where_clause', 'name': 'alacazam' },
        ],
    },
    {
        'others': [
            { 'path': 'where_clause', 'name': 'presto' },
        ],
    },
    {
        'others': [
            { 'path': 'where_clause', 'name': 'bippety' },
            { 'path': 'where_clause::Drizzel', 'name': 'boppety' },
        ],
    },
    {
        'others': [
            { 'path': 'where_clause::Drizzel', 'name': 'boppety' },
        ],
    },
];
