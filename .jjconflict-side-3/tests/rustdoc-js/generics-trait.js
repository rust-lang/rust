// exact-check

const EXPECTED = [
    {
        'query': 'Result<SomeTrait>',
        'correction': null,
        'in_args': [
            {
                'path': 'generics_trait',
                'name': 'beta',
                'displayType': '`Result`<`T`, ()> -> ()',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `SomeTrait`',
            },
        ],
        'returned': [
            {
                'path': 'generics_trait',
                'name': 'bet',
                'displayType': ' -> `Result`<`T`, ()>',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `SomeTrait`',
            },
        ],
    },
    {
        'query': 'Resulx<SomeTrait>',
        'correction': 'Result',
        'in_args': [
            {
                'path': 'generics_trait',
                'name': 'beta',
                'displayType': '`Result`<`T`, ()> -> ()',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `SomeTrait`',
            },
        ],
        'returned': [
            {
                'path': 'generics_trait',
                'name': 'bet',
                'displayType': ' -> `Result`<`T`, ()>',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `SomeTrait`',
            },
        ],
    },
    {
        'query': 'Result<SomeTraiz>',
        'proposeCorrectionFrom': 'SomeTraiz',
        'proposeCorrectionTo': 'SomeTrait',
    },
    {
        'query': 'OtherThingxxxxxxxx',
        'correction': null,
        'in_args': [
            {
                'path': 'generics_trait',
                'name': 'alpha',
                'displayType': 'Result<`T`, ()> -> ()',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `OtherThingxxxxxxxx`',
            },
        ],
        'returned': [
            {
                'path': 'generics_trait',
                'name': 'alef',
                'displayType': ' -> Result<`T`, ()>',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `OtherThingxxxxxxxx`',
            },
        ],
    },
    {
        'query': 'OtherThingxxxxxxxy',
        'correction': 'OtherThingxxxxxxxx',
        'in_args': [
            {
                'path': 'generics_trait',
                'name': 'alpha',
                'displayType': 'Result<`T`, ()> -> ()',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `OtherThingxxxxxxxx`',
            },
        ],
        'returned': [
            {
                'path': 'generics_trait',
                'name': 'alef',
                'displayType': ' -> Result<`T`, ()>',
                'displayMappedNames': '',
                'displayWhereClause': 'T: `OtherThingxxxxxxxx`',
            },
        ],
    },
];

const PARSED = [
    {
        'query': 'Result<SomeTraiz>',
        'userQuery': 'Result<SomeTraiz>',
        'foundElems': 1,
        'returned': [],
        'error': null,
        'proposeCorrectionFrom': 'SomeTraiz',
        'proposeCorrectionTo': 'SomeTrait',
    },
    {
        'query': 'Result<NoSuchTrait>',
        'userQuery': 'Result<NoSuchTrait>',
        'foundElems': 1,
        'returned': [],
        'error': null,
        'proposeCorrectionFrom': 'NoSuchTrait',
        'proposeCorrectionTo': null,
    },
];
