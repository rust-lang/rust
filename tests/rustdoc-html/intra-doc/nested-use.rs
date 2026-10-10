// Regression test for issue #113896: Intra-doc links on nested use items.

#![crate_name = "foo"]

//@ has foo/type.Foo.html
//@ has - '//a[@href="type.Foo.html"]' 'Foo'
//@ has - '//a[@href="type.Bar.html"]' 'Bar'

/// [`Foo`]
pub use m::{Foo, Bar};

mod m {
    /// [`Bar`]
    pub struct Foo;
    pub struct Bar;
}
