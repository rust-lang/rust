#![feature(custom_attribute, rustc_attrs)]
#![allow(dead_code, unused_attributes)]

#[rustc_mir]
#[miri_run]
fn foo() -> i32 {
    let x = 1;
    let y = 2;
    x + y
}

fn main() {}
