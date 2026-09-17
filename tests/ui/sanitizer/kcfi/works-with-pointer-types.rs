// Verifies that functions with pointer types as argument types can be called
// through function pointers.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

use std::ffi::c_void;

fn foo1(_: &i32) -> i32 {
    1
}
fn foo2(_: &mut i32) -> i32 {
    2
}
fn foo3(_: *const i32) -> i32 {
    3
}
fn foo4(_: *mut i32) -> i32 {
    4
}
fn foo5(_: *const c_void) -> i32 {
    5
}
fn foo6(_: *mut c_void) -> i32 {
    6
}
fn foo7(_: &&i32) -> i32 {
    7
}

fn main() {
    let mut x = 0;
    let f: fn(&i32) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(&x), 1);
    let f: fn(&mut i32) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(&mut x), 2);
    let f: fn(*const i32) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(&x as *const i32), 3);
    let f: fn(*mut i32) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f(&mut x as *mut i32), 4);
    let f: fn(*const c_void) -> i32 = std::hint::black_box(foo5);
    assert_eq!(f(&x as *const i32 as *const c_void), 5);
    let f: fn(*mut c_void) -> i32 = std::hint::black_box(foo6);
    assert_eq!(f(&mut x as *mut i32 as *mut c_void), 6);
    let f: fn(&&i32) -> i32 = std::hint::black_box(foo7);
    assert_eq!(f(&&0), 7);
}
