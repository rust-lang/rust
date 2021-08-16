#![crate_name="bar"]
#![crate_type="rlib"]
#![crate_type="cdylib"]

pub struct Bar;

pub fn bar() -> Bar {
    Bar
}
