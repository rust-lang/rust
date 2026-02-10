// exact-check

const EXPECTED = [
    {
        'query': 'macro:macro',
        'others': [
            {
                'path': 'macro_kinds',
                'name': 'macro1',
                'href': '../macro_kinds/macro.macro1.html',
                'ty': 16,
            },
            {
                'path': 'macro_kinds',
                'name': 'macro1',
                'href': '../macro_kinds/macro.macro1.html',
                'ty': 23,
            },
            {
                'path': 'macro_kinds',
                'name': 'macro1',
                'href': '../macro_kinds/macro.macro1.html',
                'ty': 24,
            },
            {
                'path': 'macro_kinds',
                'name': 'macro2',
                'href': '../macro_kinds/attr.macro2.html',
                'ty': 23,
            },
            {
                'path': 'macro_kinds',
                'name': 'macro4',
                'href': '../macro_kinds/derive.macro4.html',
                'ty': 24,
            },
            {
                'path': 'macro_kinds',
                'name': 'macro3',
                'href': '../macro_kinds/macro.macro3.html',
                'ty': 16,
            },
        ],
    },
    {
        'query': 'attr:macro',
        'others': [
            { 'path': 'macro_kinds', 'name': 'macro1', 'href': '../macro_kinds/macro.macro1.html' },
            { 'path': 'macro_kinds', 'name': 'macro2', 'href': '../macro_kinds/attr.macro2.html' },
        ],
    },
    {
        'query': 'derive:macro',
        'others': [
            { 'path': 'macro_kinds', 'name': 'macro1', 'href': '../macro_kinds/macro.macro1.html' },
            {
                'path': 'macro_kinds', 'name': 'macro4', 'href': '../macro_kinds/derive.macro4.html'
            },
        ],
    },
];
