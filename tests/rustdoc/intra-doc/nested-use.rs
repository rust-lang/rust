// Regression test for issue #113896: Intra-doc links on nested use items.

#![crate_name = "foo"]

// @has foo/struct.Foo.html
// @has - '//a[@href="struct.Foo.html"]' 'Foo'
// @has - '//a[@href="struct.Bar.html"]' 'Bar'

/// [`Foo`]
pub use m::{Foo, Bar};

mod m {
    /// [`Bar`]
    pub struct Foo;
    pub struct Bar;
}
