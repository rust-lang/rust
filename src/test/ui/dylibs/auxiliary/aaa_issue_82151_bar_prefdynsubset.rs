#![crate_name="bar"]
#![crate_type="rlib"]
#![crate_type="cdylib"]

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic=shared,std -Z prefer-dynamic-subset

pub struct Bar;

pub fn bar() -> Bar {
    Bar
}
