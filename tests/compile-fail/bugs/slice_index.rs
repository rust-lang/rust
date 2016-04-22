#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

// error-pattern:assertion failed

#[miri_run]
fn slice() -> u8 {
    let arr: &[_] = &[101, 102, 103, 104, 105, 106];
    arr[5]
}

fn main() {}
