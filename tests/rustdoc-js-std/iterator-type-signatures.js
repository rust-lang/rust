// ignore-order

const FILTER_CRATE = "std";

const EXPECTED = [
    {
        'query': 'iterator<t> -> option<t>',
        'others': [
            { 'path': 'std::iter::Iterator', 'name': 'max' },
            { 'path': 'std::iter::Iterator', 'name': 'min' },
            { 'path': 'std::iter::Iterator', 'name': 'last' },
            { 'path': 'std::iter::Iterator', 'name': 'next' },
        ],
    },
    {
        'query': 'iterator<t>, usize -> option<t>',
        'others': [
            { 'path': 'std::iter::Iterator', 'name': 'nth' },
        ],
    },
    {
        // Something should be done so that intoiterator is considered a match
        // for plain iterator.
        'query': 'iterator<t>, intoiterator<t> -> ordering',
        'others': [
            { 'path': 'std::iter::Iterator', 'name': 'cmp' },
        ],
    },
];
