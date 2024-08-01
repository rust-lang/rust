#![crate_name = "foo"]

//@ hasraw 'search.desc/foo/foo-desc-0-.js' 'Foo short link.'
//@ !hasraw - 'www.example.com'
//@ !hasraw - 'More Foo.'

/// Foo short [link](https://www.example.com/).
///
/// More Foo.
pub struct Foo;
