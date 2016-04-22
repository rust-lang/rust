#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

// error-pattern:can't handle intrinsic: size_of_val

#[miri_run]
fn memcmp() {
    assert_eq!("", "");
}

fn main() {}
