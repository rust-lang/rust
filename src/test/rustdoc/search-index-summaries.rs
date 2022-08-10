#![crate_name = "foo"]

// @hasraw 'search-index.js' 'Foo short link.'
// @!has - 'www.example.com'
// @!has - 'More Foo.'

/// Foo short [link](https://www.example.com/).
///
/// More Foo.
pub struct Foo;
