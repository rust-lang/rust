// exact-check
// Regression test for <https://github.com/rust-lang/rust/issues/150921>.

const EXPECTED = [
    {
        'query': '==',
        'others': [
            {
                'path': 'doc_alias_symbols_150921',
                'name': 'OperatorEqEqAlias',
                'alias': '==',
                'href': '../doc_alias_symbols_150921/struct.OperatorEqEqAlias.html',
                'is_alias': true,
            },
        ],
    },
    {
        'query': '!=',
        'others': [
            {
                'path': 'doc_alias_symbols_150921',
                'name': 'OperatorNotEqAlias',
                'alias': '!=',
                'href': '../doc_alias_symbols_150921/struct.OperatorNotEqAlias.html',
                'is_alias': true,
            },
        ],
    },
];
