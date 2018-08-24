// aux-build:rustdoc-extern-method.rs
// ignore-cross-compile

#![feature(unboxed_closures)]

extern crate rustdoc_extern_method as foo;

// @has extern_method/trait.Foo.html //pre "pub trait Foo"
// @has - '//*[@id="tymethod.foo"]//code' 'extern "rust-call" fn foo'
// @has - '//*[@id="method.foo_"]//code' 'extern "rust-call" fn foo_'
pub use foo::Foo;

// @has extern_method/trait.Bar.html //pre "pub trait Bar"
pub trait Bar {
    // @has - '//*[@id="tymethod.bar"]//code' 'extern "rust-call" fn bar'
    extern "rust-call" fn bar(&self, _: ());
    // @has - '//*[@id="method.bar_"]//code' 'extern "rust-call" fn bar_'
    extern "rust-call" fn bar_(&self, _: ()) { }
}
