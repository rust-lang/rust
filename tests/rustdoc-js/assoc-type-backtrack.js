// exact-check

const EXPECTED = [
    {
        'query': 'mytrait, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'fold' },
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'fold' },
        ],
    },
    {
        'query': 'mytrait<U>, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'fold' },
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'fold' },
        ],
    },
    {
        'query': 'mytrait<Item=U>, mytrait2 -> T',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyTrait', 'name': 'fold' },
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'fold' },
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
        'query': 'mytrait<U> -> Option<T>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::Cloned', 'name': 'next' },
        ],
    },
    {
        'query': 'mytrait<Item=U> -> Option<T>',
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
        'query': 'myintofuture<myfuture<t>> -> myfuture<t>',
        'correction': null,
        'others': [
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future' },
            { 'path': 'assoc_type_backtrack::MyIntoFuture', 'name': 'into_future_2' },
        ],
    },
    // Invalid unboxing of the one-argument case.
    // If you unbox one of the myfutures, you need to unbox both of them.
    {
        'query': 'myintofuture<fut=t> -> myfuture<t>',
        'correction': null,
        'others': [],
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
        'query': 'myintofuture<myfuture>, myintofuture<myfuture> -> myfuture',
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
    // Invalid unboxings of the two-argument case.
    // If you unbox one of the myfutures, you need to unbox all of them.
    {
        'query': 'myintofuture<fut=t>, myintofuture<fut=myfuture<t>> -> myfuture<t>',
        'correction': null,
        'others': [],
    },
    {
        'query': 'myintofuture<fut=myfuture<t>>, myintofuture<fut=t> -> myfuture<t>',
        'correction': null,
        'others': [],
    },
    {
        'query': 'myintofuture<fut=myfuture<t>>, myintofuture<fut=myfuture<t>> -> t',
        'correction': null,
        'others': [],
    },
    // different generics don't match up either
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
