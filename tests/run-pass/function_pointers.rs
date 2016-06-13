#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

fn f() -> i32 {
    42
}

fn return_fn_ptr() -> fn() -> i32 {
    f
}

#[miri_run]
fn call_fn_ptr() -> i32 {
    return_fn_ptr()()
}

fn main() {}
