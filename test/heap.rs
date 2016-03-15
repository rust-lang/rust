#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn make_box() -> Box<i32> {
    Box::new(42)
}
