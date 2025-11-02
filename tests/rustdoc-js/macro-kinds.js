// exact-check

const EXPECTED = [
    {
        'query': 'macro:macro',
        'others': [
            { 'path': 'macro_kinds', 'name': 'macro1', 'href': '../macro_kinds/macro.macro1.html' },
            { 'path': 'macro_kinds', 'name': 'macro3', 'href': '../macro_kinds/macro.macro3.html' },
        ],
    },
    {
        'query': 'attr:macro',
        'others': [
            { 'path': 'macro_kinds', 'name': 'macro1', 'href': '../macro_kinds/attr.macro1.html' },
            { 'path': 'macro_kinds', 'name': 'macro2', 'href': '../macro_kinds/attr.macro2.html' },
            { 'path': 'macro_kinds', 'name': 'macro5', 'href': '../macro_kinds/attr.macro5.html' },
        ],
    },
    {
        'query': 'derive:macro',
        'others': [
            {
                'path': 'macro_kinds', 'name': 'macro1', 'href': '../macro_kinds/derive.macro1.html'
            },
            {
                'path': 'macro_kinds', 'name': 'macro5', 'href': '../macro_kinds/derive.macro5.html'
            },
            {
                'path': 'macro_kinds', 'name': 'macro4', 'href': '../macro_kinds/derive.macro4.html'
            },
        ],
    },
];
