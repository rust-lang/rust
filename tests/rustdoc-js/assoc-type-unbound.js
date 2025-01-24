// exact-check

const EXPECTED = [
    // Trait-associated types (that is, associated types with no constraints)
    // are treated like type parameters, so that you can "pattern match"
    // them. We should avoid redundant output (no `Item=MyIter::Item` stuff)
    // and should give reasonable results
    {
        'query': 'MyIter<T> -> Option<T>',
        'correction': null,
        'others': [
            {
                'path': 'assoc_type_unbound::MyIter',
                'name': 'next',
                'displayType': '&mut `MyIter` -> `Option`<`MyIter::Item`>',
                'displayMappedNames': 'T = MyIter::Item',
                'displayWhereClause': '',
            },
        ],
    },
    {
        'query': 'MyIter<Item=T> -> Option<T>',
        'correction': null,
        'others': [
            {
                'path': 'assoc_type_unbound::MyIter',
                'name': 'next',
                'displayType': '&mut `MyIter` -> `Option`<`MyIter::Item`>',
                'displayMappedNames': 'T = MyIter::Item',
                'displayWhereClause': '',
            },
        ],
    },
    {
        'query': 'MyIter<T> -> Option<Item=T>',
        'correction': null,
        'others': [],
    },
];
