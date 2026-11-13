#![crate_name = "foo"]

pub struct Foo;

//@ has foo/struct.Bar.html '//a[@href="struct.Foo.html"]' 'Foo'

/// Code-styled reference to [`Foo`].
pub struct Bar;
