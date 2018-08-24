#![crate_name = "foo"]

// @has 'search-index.js' 'Foo short link.'
// @!has - 'www.example.com'
// @!has - 'More Foo.'

/// Foo short [link](https://www.example.com/).
///
/// More Foo.
pub struct Foo;
