#![crate_name = "foo"]

// https://github.com/rust-lang/rust/issues/34928

pub trait Bar {}

//@ has foo/struct.Foo.html '//pre' 'pub struct Foo<T>(pub T) where T: Bar;'
pub struct Foo<T>(pub T) where T: Bar;
