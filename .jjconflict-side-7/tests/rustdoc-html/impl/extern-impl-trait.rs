//@ aux-build:extern-impl-trait.rs

#![crate_name = "foo"]

extern crate extern_impl_trait;

//@ has 'foo/struct.X.html' '//h4[@class="code-header"]' "impl Foo<Associated = ()> + 'a"
pub use extern_impl_trait::X;

//@ has 'foo/struct.Y.html' '//h4[@class="code-header"]' "impl Foo<Associated = ()> + ?Sized + 'a"
pub use extern_impl_trait::Y;
