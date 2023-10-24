// exact-check
// ignore-order

const EXPECTED = [
    {
        query: '-> trait:Some',
        others: [
            { path: 'foo', name: 'alef', displayTypeSignature: '-> impl *Some*' },
            { path: 'foo', name: 'alpha', displayTypeSignature: '-> impl *Some*' },
        ],
    },
    {
        query: '-> generic:T',
        others: [
            { path: 'foo', name: 'bet', displayTypeSignature: '_ -> *T*' },
            { path: 'foo', name: 'alef', displayTypeSignature: '-> *T*' },
            { path: 'foo', name: 'beta', displayTypeSignature: 'T -> *T*' },
        ],
    },
    {
        query: 'A -> B',
        others: [
            { path: 'foo', name: 'bet', displayTypeSignature: '*A* -> *B*' },
        ],
    },
    {
        query: 'A -> A',
        others: [
            { path: 'foo', name: 'beta', displayTypeSignature: '*A* -> *A*' },
        ],
    },
    {
        query: 'A, A',
        others: [
            { path: 'foo', name: 'alternate', displayTypeSignature: '*A*, *A*' },
        ],
    },
    {
        query: 'A, B',
        others: [
            { path: 'foo', name: 'other', displayTypeSignature: '*A*, *B*' },
        ],
    },
    {
        query: 'Other, Other',
        others: [
            { path: 'foo', name: 'other', displayTypeSignature: 'impl *Other*, impl *Other*' },
            { path: 'foo', name: 'alternate', displayTypeSignature: 'impl *Other*, impl *Other*' },
        ],
    },
    {
        query: 'generic:T',
        in_args: [
            { path: 'foo', name: 'bet', displayTypeSignature: '*T* -> _' },
            { path: 'foo', name: 'beta', displayTypeSignature: '*T* -> T' },
            { path: 'foo', name: 'other', displayTypeSignature: '*T*, _' },
            { path: 'foo', name: 'alternate', displayTypeSignature: '*T*, T' },
        ],
    },
    {
        query: 'generic:Other',
        in_args: [
            { path: 'foo', name: 'bet', displayTypeSignature: '*Other* -> _' },
            { path: 'foo', name: 'beta', displayTypeSignature: '*Other* -> Other' },
            { path: 'foo', name: 'other', displayTypeSignature: '*Other*, _' },
            { path: 'foo', name: 'alternate', displayTypeSignature: '*Other*, Other' },
        ],
    },
    {
        query: 'trait:Other',
        in_args: [
            { path: 'foo', name: 'other', displayTypeSignature: '_, impl *Other*' },
            { path: 'foo', name: 'alternate', displayTypeSignature: 'impl *Other*, impl *Other*' },
        ],
    },
    {
        query: 'Other',
        in_args: [
            { path: 'foo', name: 'other', displayTypeSignature: '_, impl *Other*' },
            { path: 'foo', name: 'alternate', displayTypeSignature: 'impl *Other*, impl *Other*' },
        ],
    },
    {
        query: 'trait:T',
        in_args: [],
    },
];
