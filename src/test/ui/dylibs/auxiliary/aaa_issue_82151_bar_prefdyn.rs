#![crate_name="bar"]
#![crate_type="rlib"]
#![crate_type="cdylib"]

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic

pub struct Bar;

pub fn bar() -> Bar {
    Bar
}
