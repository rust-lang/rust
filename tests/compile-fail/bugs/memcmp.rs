#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

// error-pattern:can't call C ABI function: memcmp

#[miri_run]
fn memcmp() {
    assert_eq!("", "");
}

fn main() {}
