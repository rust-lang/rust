#![crate_name = "foo"]

pub struct Foo;

//@ has foo/type.Bar.html '//a[@href="type.Foo.html"]' 'Foo'

/// Code-styled reference to [`Foo`].
pub struct Bar;
