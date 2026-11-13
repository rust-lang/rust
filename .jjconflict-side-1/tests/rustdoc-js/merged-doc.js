const EXPECTED = [
    {
        'query': 'merged_doc::Doc',
        'others': [
            { 'path': 'merged_doc', 'name': 'Doc' },
        ],
    },
    {
        'query': 'merged_dep::Dep',
        'others': REVISION === "nomerge" ? [] :
            [
                { 'path': 'merged_dep', 'name': 'Dep' },
            ],
    },
];
