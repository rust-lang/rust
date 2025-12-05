// exact-check

const EXPECTED = [
    {
        'query': 'mytrait<t> -> option<t>',
        'correction': null,
        'in_args': [],
        'others': [
            { 'path': 'trait_methods::MyTrait', 'name': 'next' },
        ],
    },
    // the traitParent deduplication pass should remove
    // Empty::next, as it would be redundant
    {
        'query': 'next',
        'correction': null,
        'in_args': [],
        'others': [
            { 'path': 'trait_methods::MyTrait', 'name': 'next' },
        ],
    },
    // if the trait does not match, no deduplication happens
    {
        'query': '-> option<()>',
        'correction': null,
        'in_args': [],
        'others': [
            { 'path': 'trait_methods::Empty', 'name': 'next' },
                    { 'path': 'trait_methods::Void', 'name': 'next' },
        ],
    },
];
