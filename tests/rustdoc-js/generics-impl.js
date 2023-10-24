// exact-check

const EXPECTED = [
    {
        'query': 'Aaaaaaa -> u32',
        'others': [
            {
                'path': 'generics_impl::Aaaaaaa',
                'name': 'bbbbbbb',
                'displayTypeSignature': '*Aaaaaaa* -> *u32*'
            },
        ],
    },
    {
        'query': 'Aaaaaaa -> bool',
        'others': [
            {
                'path': 'generics_impl::Aaaaaaa',
                'name': 'ccccccc',
                'displayTypeSignature': '*Aaaaaaa* -> *bool*'
            },
        ],
    },
    {
        'query': 'Aaaaaaa -> usize',
        'others': [
            {
                'path': 'generics_impl::Aaaaaaa',
                'name': 'read',
                'displayTypeSignature': '*Aaaaaaa*, [] -> Result<*usize*>'
            },
        ],
    },
    {
        'query': 'Read -> u64',
        'others': [
            {
                'path': 'generics_impl::Ddddddd',
                'name': 'eeeeeee',
                'displayTypeSignature': 'impl *Read* -> *u64*'
            },
            {
                'path': 'generics_impl::Ddddddd',
                'name': 'ggggggg',
                'displayTypeSignature': 'Ddddddd<impl *Read*> -> *u64*'
            },
        ],
    },
    {
        'query': 'trait:Read -> u64',
        'others': [
            {
                'path': 'generics_impl::Ddddddd',
                'name': 'eeeeeee',
                'displayTypeSignature': 'impl *Read* -> *u64*'
            },
            {
                'path': 'generics_impl::Ddddddd',
                'name': 'ggggggg',
                'displayTypeSignature': 'Ddddddd<impl *Read*> -> *u64*'
            },
        ],
    },
    {
        'query': 'struct:Read -> u64',
        'others': [],
    },
    {
        'query': 'bool -> u64',
        'others': [
            {
                'path': 'generics_impl::Ddddddd',
                'name': 'fffffff',
                'displayTypeSignature': '*bool* -> *u64*'
            },
        ],
    },
    {
        'query': 'Ddddddd -> u64',
        'others': [
            {
                'path': 'generics_impl::Ddddddd',
                'name': 'ggggggg',
                'displayTypeSignature': '*Ddddddd* -> *u64*'
            },
        ],
    },
    {
        'query': '-> Ddddddd',
        'others': [
            {
                'path': 'generics_impl::Ddddddd',
                'name': 'hhhhhhh',
                'displayTypeSignature': '-> *Ddddddd*'
            },
        ],
    },
];
