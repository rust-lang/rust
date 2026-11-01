// Verifies that functions with pattern types as argument types can be called
// through function pointers.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

fn foo1(x: pattern_type!(i32 is 1..)) -> i32 {
    assert_eq!(unsafe { std::mem::transmute::<_, i32>(x) }, 1);
    1
}

fn foo2(x: pattern_type!(i32 is 1..=5)) -> i32 {
    assert_eq!(unsafe { std::mem::transmute::<_, i32>(x) }, 2);
    2
}

fn foo3(x: pattern_type!(i32 is -5..=5)) -> i32 {
    assert_eq!(unsafe { std::mem::transmute::<_, i32>(x) }, -3);
    3
}

fn main() {
    let x: pattern_type!(i32 is 1..) = unsafe { std::mem::transmute(1i32) };
    let f: fn(pattern_type!(i32 is 1..)) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(x), 1);
    let x: pattern_type!(i32 is 1..=5) = unsafe { std::mem::transmute(2i32) };
    let f: fn(pattern_type!(i32 is 1..=5)) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(x), 2);
    let x: pattern_type!(i32 is -5..=5) = unsafe { std::mem::transmute(-3i32) };
    let f: fn(pattern_type!(i32 is -5..=5)) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(x), 3);
}
