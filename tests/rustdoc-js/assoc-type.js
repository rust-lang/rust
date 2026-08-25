// exact-check

const EXPECTED = [
    // if I just use generics, then the generics version
    // and the type binding version both show up
    {
        'query': 'iterator<something> -> u32',
        'correction': null,
        'others': [
            {
                'path': 'assoc_type::my',
                'name': 'other_fn',
                'displayType': 'X -> `u32`',
                'displayMappedNames': '',
                'displayWhereClause': 'X: `Iterator`<`Something`>',
            },
            {
                'path': 'assoc_type',
                'name': 'my_fn',
                'displayType': 'X -> `u32`',
                'displayMappedNames': '',
                'displayWhereClause': 'X: `Iterator`<Item=`Something`>',
            },
        ],
    },
    {
        'query': 'iterator<something>',
        'correction': null,
        'in_args': [
            {
                'path': 'assoc_type::my',
                'name': 'other_fn',
                'displayType': 'X -> u32',
                'displayMappedNames': '',
                'displayWhereClause': 'X: `Iterator`<`Something`>',
            },
            {
                'path': 'assoc_type',
                'name': 'my_fn',
                'displayType': 'X -> u32',
                'displayMappedNames': '',
                'displayWhereClause': 'X: `Iterator`<Item=`Something`>',
            },
        ],
    },
    {
        'query': 'something',
        'correction': null,
        'others': [
            { 'path': 'assoc_type', 'name': 'Something' },
        ],
        'in_args': [
            {
                'path': 'assoc_type::my',
                'name': 'other_fn',
                'displayType': '`X` -> u32',
                'displayMappedNames': '',
                'displayWhereClause': 'X: Iterator<`Something`>',
            },
            {
                'path': 'assoc_type',
                'name': 'my_fn',
                'displayType': '`X` -> u32',
                'displayMappedNames': '',
                'displayWhereClause': 'X: Iterator<Item=`Something`>',
            },
        ],
    },
    // if I write an explicit binding, only it shows up
    {
        'query': 'iterator<item=something> -> u32',
        'correction': null,
        'others': [
            { 'path': 'assoc_type', 'name': 'my_fn' },
        ],
    },
    // case insensitivity
    {
        'query': 'iterator<ItEm=sOmEtHiNg> -> u32',
        'correction': null,
        'others': [
            { 'path': 'assoc_type', 'name': 'my_fn' },
        ],
    },
    // wrong binding name, no result
    {
        'query': 'iterator<something=something> -> u32',
        'correction': null,
        'in_args': [],
        'others': [],
    },
];
