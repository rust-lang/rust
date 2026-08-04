// Verifies that functions with sequence types as argument types can be called
// through function pointers.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

fn foo1(x: (i32, i32)) -> i32 {
    assert_eq!(x, (1, 2));
    1
}

fn foo2(x: [i32; 4]) -> i32 {
    assert_eq!(x, [1, 2, 3, 4]);
    2
}

fn foo3(x: &[i32]) -> i32 {
    assert_eq!(x, &[1, 2, 3]);
    3
}

fn foo4(x: &str) -> i32 {
    assert_eq!(x, "foo");
    4
}

fn foo5(x: [i32; 2 * 2]) -> i32 {
    assert_eq!(x, [1, 2, 3, 4]);
    5
}

fn main() {
    let f: fn((i32, i32)) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f((1, 2)), 1);
    let f: fn([i32; 4]) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f([1, 2, 3, 4]), 2);
    let f: fn(&[i32]) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(&[1, 2, 3]), 3);
    let f: fn(&str) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f("foo"), 4);
    let f: fn([i32; 2 * 2]) -> i32 = std::hint::black_box(foo5);
    assert_eq!(f([1, 2, 3, 4]), 5);
}
