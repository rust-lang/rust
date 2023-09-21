// exact-check
// ignore-order

const EXPECTED = [
    {
        query: '-> trait:Some',
        others: [
            { path: 'foo', name: 'alef' },
            { path: 'foo', name: 'alpha' },
        ],
    },
    {
        query: '-> generic:T',
        others: [
            { path: 'foo', name: 'bet' },
            { path: 'foo', name: 'alef' },
            { path: 'foo', name: 'beta' },
        ],
    },
    {
        query: 'A -> B',
        others: [
            { path: 'foo', name: 'bet' },
        ],
    },
    {
        query: 'A -> A',
        others: [
            { path: 'foo', name: 'beta' },
        ],
    },
    {
        query: 'A, A',
        others: [
            { path: 'foo', name: 'alternate' },
        ],
    },
    {
        query: 'A, B',
        others: [
            { path: 'foo', name: 'other' },
        ],
    },
    {
        query: 'Other, Other',
        others: [
            { path: 'foo', name: 'other' },
            { path: 'foo', name: 'alternate' },
        ],
    },
    {
        query: 'generic:T',
        in_args: [
            { path: 'foo', name: 'bet' },
            { path: 'foo', name: 'beta' },
            { path: 'foo', name: 'other' },
            { path: 'foo', name: 'alternate' },
        ],
    },
    {
        query: 'generic:Other',
        in_args: [
            { path: 'foo', name: 'bet' },
            { path: 'foo', name: 'beta' },
            { path: 'foo', name: 'other' },
            { path: 'foo', name: 'alternate' },
        ],
    },
    {
        query: 'trait:Other',
        in_args: [
            { path: 'foo', name: 'other' },
            { path: 'foo', name: 'alternate' },
        ],
    },
    {
        query: 'Other',
        in_args: [
            { path: 'foo', name: 'other' },
            { path: 'foo', name: 'alternate' },
        ],
    },
    {
        query: 'trait:T',
        in_args: [],
    },
];
