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

#[miri_run]
fn call_generic() -> (i16, bool) {
    fn id<T>(t: T) -> T { t }
    (id(42), id(true))
}

// Test calling a very simple function from the standard library.
#[miri_run]
fn cross_crate_fn_call() -> i64 {
    if 1i32.is_positive() { 1 } else { 0 }
}

// Test one of the simplest intrinsics.
#[miri_run]
fn test_size_of() -> usize {
    ::std::mem::size_of::<Option<i32>>()
}

#[miri_run]
fn main() {
    assert_eq!(call(), 2);
    assert_eq!(factorial_recursive(), 3628800);
    //assert_eq!(call_generic(), (42, true));
    assert_eq!(cross_crate_fn_call(), 1);
    //assert_eq!(test_size_of(), 8);
}
