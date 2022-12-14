#![crate_name = "foo"]

pub trait Bar {}

// @has foo/struct.Foo.html '//pre' 'pub struct Foo<T>(pub T)where T: Bar;'
pub struct Foo<T>(pub T) where T: Bar;
