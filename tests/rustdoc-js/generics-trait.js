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
        'in_args': [],
        'returned': [],
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
