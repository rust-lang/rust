const QUERY = ['trait<nested>', '-> trait<nested>', 't1, t2'];

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
];
