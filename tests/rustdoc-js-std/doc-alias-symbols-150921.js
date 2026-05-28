// exact-check
// Regression test for <https://github.com/rust-lang/rust/issues/150921>.

const EXPECTED = [
    {
        'query': '==',
        'others': [
            {
                'path': 'std::cmp',
                'name': 'Eq',
                'alias': '==',
                'href': '../std/cmp/trait.Eq.html',
                'is_alias': true,
            },
            {
                'path': 'std::cmp',
                'name': 'PartialEq',
                'alias': '==',
                'href': '../std/cmp/trait.PartialEq.html',
                'is_alias': true,
            },
        ],
    },
];
