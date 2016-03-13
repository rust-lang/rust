#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn call() -> i32 {
    fn increment(x: i32) -> i32 {
        x + 1
    }
    increment(1)
}

#[miri_run]
fn factorial_recursive() -> i64 {
    fn fact(n: i64) -> i64 {
        if n == 0 {
            1
        } else {
            n * fact(n - 1)
        }
    }
    fact(10)
}

// Test calling a very simple function from the standard library.
#[miri_run]
fn cross_crate_fn_call() -> i64 {
    if 1i32.is_positive() { 1 } else { 0 }
}
