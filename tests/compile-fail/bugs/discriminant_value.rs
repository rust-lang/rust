#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

// error-pattern:can't handle intrinsic: discriminant_value

#[miri_run]
fn main() {
    assert_eq!(None::<i32>, None);
}
