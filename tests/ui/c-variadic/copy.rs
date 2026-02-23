//@ run-pass
//@ ignore-backends: gcc
#![feature(c_variadic)]

// Test the behavior of `VaList::clone`. In C a `va_list` is duplicated using `va_copy`, but the
// rust api just uses `Clone`. This should create a completely independent cursor into the
// variable argument list: advancing the original has no effect on the copy and vice versa.

fn main() {
    unsafe { variadic(1, 2, 3) }
}

unsafe extern "C" fn variadic(mut ap1: ...) {
    let mut ap2 = ap1.clone();

    assert_eq!(ap1.arg::<i32>(), 1);
    assert_eq!(ap2.arg::<i32>(), 1);

    assert_eq!(ap2.arg::<i32>(), 2);
    assert_eq!(ap1.arg::<i32>(), 2);

    drop(ap1);
    assert_eq!(ap2.arg::<i32>(), 3);
}
