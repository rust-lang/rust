// aux-build:extern-impl-trait.rs

#![crate_name = "foo"]

extern crate extern_impl_trait;

// @has 'foo/struct.X.html' '//code' "impl Foo<Associated = ()> + 'a"
pub use extern_impl_trait::X;

// @has 'foo/struct.Y.html' '//code' "impl ?Sized + Foo<Associated = ()> + 'a"
pub use extern_impl_trait::Y;
