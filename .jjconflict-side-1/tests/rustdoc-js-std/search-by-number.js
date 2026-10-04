// regression test for https://github.com/rust-lang/rust/issues/147763
//
// identifiers in search queries should not be required to follow the
// same strict rules around ID_Start that identifers in rust code follow,
// as searches frequently use substrings of identifers.
//
// for example, identifiers cannot start with digits,
// but they can contain them, so we allow search idents to start with digits.

const EXPECTED = {
    'query': '8',
    'others': [
        {
            'path': 'std',
            'name': 'i8',
            'href': '../std/primitive.i8.html',
        },
    ]
};
