// exact-check

const FILTER_CRATE = 'doc_alias_filter';

const EXPECTED = {
    'query': '"true"',
    'others': [
        {
            'path': 'doc_alias_filter',
            'name': 'Foo',
            'alias': 'true',
            'href': '../doc_alias_filter/struct.Foo.html',
            'is_alias': true
        },
    ],
};
