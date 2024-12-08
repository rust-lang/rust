// exact-check

const EXPECTED = [
    {
        'query': 'outside<U>, outside<V> -> outside<W>',
        'others': [],
    },
    {
        'query': 'outside<V>, outside<U> -> outside<W>',
        'others': [],
    },
    {
        'query': 'outside<U>, outside<U> -> outside<W>',
        'others': [],
    },
    {
        'query': 'outside<U>, outside<U> -> outside<U>',
        'others': [
            {"path": "generics2", "name": "should_match_3"}
        ],
    },
];
