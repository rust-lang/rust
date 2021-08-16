#![crate_name="foo"]
#![crate_type="rlib"]

// no-prefer-dynamic

extern crate bar;

pub struct Foo;
pub use bar::bar;

pub fn foo() -> Foo {
    Foo
}
