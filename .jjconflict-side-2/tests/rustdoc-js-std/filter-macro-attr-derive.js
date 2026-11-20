// This test ensures that filtering on "macro" will also include attribute and derive
// macros.

const EXPECTED = {
    'query': 'macro:debug',
    'others': [
        { 'path': 'std::fmt', 'name': 'Debug', 'href': '../std/fmt/derive.Debug.html' },
    ],
};
