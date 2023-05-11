#![crate_name = "foo"]

// @hasraw 'search-index.js' 'Foo short link.'
// @!hasraw - 'www.example.com'
// @!hasraw - 'More Foo.'

/// Foo short [link](https://www.example.com/).
///
/// More Foo.
pub struct Foo;
