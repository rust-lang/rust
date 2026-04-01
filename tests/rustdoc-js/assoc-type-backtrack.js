// exact-check

const EXPECTED = [
    {
        'query': 'mytrait, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'fold' },
        ],
    },
    {
        'query': 'mytrait<U>, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'fold' },
        ],
    },
    {
        'query': 'cloned<mytrait>, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'fold' },
        ],
    },
    {
        'query': 'cloned<mytrait<U>>, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'fold' },
        ],
    },
    {
        'query': 'mytrait<Item=U>, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'fold' },
        ],
    },
    {
        'query': 'mytrait<T>, mytrait2 -> T',
        'correction': null,
        'others': [],
    },
    {
        'query': 'mytrait<Item=T>, mytrait2 -> T',
        'correction': null,
        'others': [],
    },
    {
        'query': 'mytrait<T> -> Option<T>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'next' },
        ],
    },
    {
        'query': 'mytrait<Item=T> -> Option<T>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'next' },
        ],
    },
    {
        'query': 'cloned<mytrait<U>> -> Option<T>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'next' },
        ],
    },
    {
        'query': 'cloned<mytrait<Item=U>> -> Option<T>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'next' },
        ],
    },
    // The first two define the base case.
    {
        'query': 'myintofuture<fut=myfuture<t>> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future' },
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    {
        'query': 'myintofuture<fut=myfuture<t>>, myintofuture<fut=myfuture<t>> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    // Unboxings of the one-argument case.
    {
        'query': 'myfuture<t> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future' },
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    {
        'query': 'myintofuture<t, myfuture<t>> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future' },
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    // Unboxings of the one-argument case.
    {
        'query': 'myintofuture<fut=t> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future' },
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    // Unboxings of the two-argument case.
    {
        'query': 'myintofuture<fut=t>, myintofuture<fut=t> -> t',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    {
        'query': 'myintofuture<fut=myfuture>, myintofuture<fut=myfuture> -> myfuture',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    {
        'query': 'myintofuture<t, myfuture>, myintofuture<t, myfuture> -> myfuture',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    {
        'query': 'myfuture<t>, myfuture<t> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    // If you unbox one of the myfutures, you don't need to unbox all of them.
    {
        'query': 'myintofuture<fut=t>, myintofuture<fut=myfuture<t>> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    {
        'query': 'myintofuture<fut=myfuture<t>>, myintofuture<fut=t> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    {
        'query': 'myintofuture<fut=myfuture<t>>, myintofuture<fut=myfuture<t>> -> t',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    // different generics must match up
    {
        'query': 'myintofuture<fut=myfuture<u>>, myintofuture<fut=myfuture<t>> -> myfuture<t>',
        'correction': null,
        'others': [],
    },
    {
        'query': 'myintofuture<output=t> -> myfuture<tt>',
        'correction': null,
        'others': [],
    },
];
