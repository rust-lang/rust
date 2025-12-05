const EXPECTED = [
    {
        'query': 'T -> T',
        'correction': null,
        'others': [
            {
                'path': 'enum_variant_not_type',
                'name': 'my_fn',
            },
            {
                'path': 'enum_variant_not_type::AutoCorrectConfounder',
                'name': 'assoc_type_acts_like_generic',
            },
        ],
    },
    {
        'query': 'InsertUnnecessarilyLongTypeNameHere -> InsertUnnecessarilyLongTypeNameHere',
        'correction': null,
        'others': [
            {
                'path': 'enum_variant_not_type',
                'name': 'my_fn',
            },
            {
                'path': 'enum_variant_not_type::AutoCorrectConfounder',
                'name': 'assoc_type_acts_like_generic',
            },
        ],
    },
    {
        'query': 'InsertUnnecessarilyLongTypeNameHere',
        'correction': null,
        'others': [
            {
                'path': 'enum_variant_not_type::AutoCorrectConfounder',
                'name': 'InsertUnnecessarilyLongTypeNameHere',
            },
        ],
    },
    {
        'query': 'InsertUnnecessarilyLongTypeNameHereX',
        'correction': null,
        'others': [
            {
                'path': 'enum_variant_not_type::AutoCorrectConfounder',
                'name': 'InsertUnnecessarilyLongTypeNameHere',
            },
        ],
    },
    {
        'query': 'T',
        'correction': null,
        'others': [
            {
                'path': 'enum_variant_not_type::MyTrait',
                'name': 'T',
            },
        ],
    },
    {
        'query': 'T',
        'correction': null,
        'others': [
            {
                'path': 'enum_variant_not_type::MyTrait',
                'name': 'T',
            },
        ],
    },
];
