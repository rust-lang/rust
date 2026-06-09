// ignore-order
// exact-check

// Make sure that results are order-agnostic, even when there's search items that only differ
// by generics.

const EXPECTED = [
    {
        'query': 'Wrap',
        'in_args': [
            { 'path': 'generics_match_ambiguity', 'name': 'bar' },
            { 'path': 'generics_match_ambiguity', 'name': 'foo' },
        ],
    },
    {
        'query': 'Wrap<i32>',
        'in_args': [
            { 'path': 'generics_match_ambiguity', 'name': 'bar' },
            { 'path': 'generics_match_ambiguity', 'name': 'foo' },
        ],
    },
    {
        'query': 'Wrap<i32>, Wrap<i32, u32>',
        'others': [
            { 'path': 'generics_match_ambiguity', 'name': 'bar' },
            { 'path': 'generics_match_ambiguity', 'name': 'foo' },
        ],
    },
    {
        'query': 'Wrap<i32, u32>, Wrap<i32>',
        'others': [
            { 'path': 'generics_match_ambiguity', 'name': 'bar' },
            { 'path': 'generics_match_ambiguity', 'name': 'foo' },
        ],
    },
    {
        'query': 'W3<i32>, W3<i32, u32>',
        'others': [
            { 'path': 'generics_match_ambiguity', 'name': 'baaa' },
            { 'path': 'generics_match_ambiguity', 'name': 'baab' },
        ],
    },
    {
        'query': 'W3<i32, u32>, W3<i32>',
        'others': [
            { 'path': 'generics_match_ambiguity', 'name': 'baaa' },
            { 'path': 'generics_match_ambiguity', 'name': 'baab' },
        ],
    },
    {
        // strict generics matching; W2<i32, u32> doesn't match W2<W3<i32, u32>>,
        // even though W2<i32> works just fine (ignoring the W3)
        'query': 'W2<i32>, W2<i32, u32>',
        'others': [],
    },
    {
        'query': 'W2<i32, u32>, W2<i32>',
        'others': [],
    },
    {
        'query': 'W2<i32>, W3<i32, u32>',
        'others': [],
    },
    {
        'query': 'W2<i32>, W2<i32>',
        'others': [],
    },
];
