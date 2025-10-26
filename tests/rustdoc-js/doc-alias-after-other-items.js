// exact-check

// Checking that doc aliases are always listed after items with equivalent matching.

const EXPECTED = [
    {
        'query': 'coo',
        'others': [
            {
                'path': 'doc_alias_after_other_items',
                'name': 'Foo',
                'href': '../doc_alias_after_other_items/struct.Foo.html',
            },
            {
                'path': 'doc_alias_after_other_items',
                'name': 'bar',
                'alias': 'Boo',
                'is_alias': true
            },
        ],
    },
    {
        'query': '"confiture"',
        'others': [
            {
                'path': 'doc_alias_after_other_items',
                'name': 'Confiture',
                'href': '../doc_alias_after_other_items/struct.Confiture.html',
            },
            {
                'path': 'doc_alias_after_other_items',
                'name': 'this_is_a_long_name',
                'alias': 'Confiture',
                'is_alias': true
            },
        ],
    },
];
