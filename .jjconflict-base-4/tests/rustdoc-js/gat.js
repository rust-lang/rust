// exact-check

const EXPECTED = [
    {
        'query': 'foo<assoc<u8>=u8> -> u32',
        'correction': null,
        'in_args': [],
        'others': [
            { 'path': 'gat', 'name': 'sample' },
        ],
    },
    {
        'query': 'foo<assoc<u8>=u8> -> !',
        'correction': null,
        'in_args': [],
        'others': [
            { 'path': 'gat', 'name': 'synergy' },
        ],
    },
    {
        'query': 'foo<assoc<u8>=u8>',
        'correction': null,
        'in_args': [
            { 'path': 'gat', 'name': 'sample' },
            { 'path': 'gat', 'name': 'synergy' },
        ],
    },
    {
        'query': 'foo<assoc<u8>=u32>',
        'correction': null,
        'in_args': [
            { 'path': 'gat', 'name': 'consider' },
        ],
    },
    {
        // This one is arguably a bug, because the way rustdoc
        // stores GATs in the search index is sloppy, but it's
        // precise enough to match most of the samples in the
        // GAT initiative repo
        'query': 'foo<assoc<u32>=u8>',
        'correction': null,
        'in_args': [
            { 'path': 'gat', 'name': 'consider' },
        ],
    },
    {
        // This one is arguably a bug, because the way rustdoc
        // stores GATs in the search index is sloppy, but it's
        // precise enough to match most of the samples in the
        // GAT initiative repo
        'query': 'foo<assoc<T>=T>',
        'correction': null,
        'in_args': [
            { 'path': 'gat', 'name': 'integrate' },
        ],
    },
];
