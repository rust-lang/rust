//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@[edition2024] compile-flags: -Zunstable-options
//@ check-pass

#![feature(unsafe_extern_blocks)]

unsafe extern "C" {
    safe static TEST1: i32;
    safe fn test1(i: i32);
}

fn test2() {
    test1(TEST1);
}

fn main() {}
