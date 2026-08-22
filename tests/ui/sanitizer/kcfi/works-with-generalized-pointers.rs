// Verifies that functions that differ only in the pointee types of their
// pointer arguments can be called through function pointers when compiling with
// -Zsanitizer-cfi-generalize-pointers.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Zsanitizer-cfi-generalize-pointers -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

fn foo1(_: *const i32) -> i32 {
    1
}

fn foo2(_: *mut i32) -> i32 {
    2
}

fn foo3(_: &i32) -> i32 {
    3
}

fn foo4(_: &mut i32) -> i32 {
    4
}

fn foo5(_: fn(i32) -> i32) -> i32 {
    5
}

fn main() {
    // Pointers and references are generalized to *const (), so the type ids encoded for the
    // functions above and for the fn pointer types below are the same and match.
    let mut x = 0;
    let f: fn(*const i32) -> i32 = std::hint::black_box(foo1);
    let f: fn(*const i8) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(&x as *const i32 as *const i8), 1);
    let f: fn(*mut i32) -> i32 = std::hint::black_box(foo2);
    let f: fn(*mut i8) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(&mut x as *mut i32 as *mut i8), 2);
    let f: fn(&i32) -> i32 = std::hint::black_box(foo3);
    let f: fn(&i8) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(unsafe { &*(&x as *const i32 as *const i8) }), 3);
    let f: fn(&mut i32) -> i32 = std::hint::black_box(foo4);
    let f: fn(&mut i8) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(unsafe { &mut *(&mut x as *mut i32 as *mut i8) }), 4);

    // Function pointers are generalized to *const () as well
    let f: fn(fn(i32) -> i32) -> i32 = std::hint::black_box(foo5);
    let f: fn(fn(u64) -> u64) -> i32 = std::hint::black_box(unsafe { std::mem::transmute(f) });
    assert_eq!(f(|x| x), 5);
}
