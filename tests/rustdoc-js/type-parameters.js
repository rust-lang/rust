// exact-check

const EXPECTED = [
    {
        query: '-> trait:Some',
        others: [
            { path: 'foo', name: 'alpha' },
            { path: 'foo', name: 'alef' },
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
            { path: 'foo', name: 'alternate' },
            { path: 'foo', name: 'other' },
        ],
    },
    {
        query: 'generic:T',
        in_args: [
            { path: 'foo', name: 'bet' },
            { path: 'foo', name: 'beta' },
            { path: 'foo', name: 'alternate' },
            { path: 'foo', name: 'other' },
        ],
    },
    {
        query: 'generic:Other',
        in_args: [
            { path: 'foo', name: 'bet' },
            { path: 'foo', name: 'beta' },
            { path: 'foo', name: 'alternate' },
            { path: 'foo', name: 'other' },
        ],
    },
    {
        query: 'trait:Other',
        in_args: [
            { path: 'foo', name: 'alternate' },
            { path: 'foo', name: 'other' },
        ],
    },
    {
        query: 'Other',
        in_args: [
            // because function is called "other", it's sorted first
            // even though it has higher type distance
            { path: 'foo', name: 'other' },
            { path: 'foo', name: 'alternate' },
        ],
    },
    {
        query: 'trait:T',
        in_args: [],
    },
];
