//@ revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@[edition2024] compile-flags: -Zunstable-options
//@ check-pass

unsafe extern "C" {
    safe fn test1(i: i32);
}

fn test2(i: i32) {
    test1(i);
}

fn main() {}
