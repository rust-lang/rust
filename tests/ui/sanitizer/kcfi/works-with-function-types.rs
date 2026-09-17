// Verifies that functions with function types (i.e., function pointers) as
// argument types can be called through function pointers.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

fn bar1(x: i32) -> i32 {
    x
}

unsafe fn bar2() {}

extern "C" fn bar3() {}

fn foo1(f: fn(i32) -> i32) -> i32 {
    assert_eq!(f(1), 1);
    1
}

fn foo2(f: unsafe fn()) -> i32 {
    unsafe { f() };
    2
}

fn foo3(f: extern "C" fn()) -> i32 {
    f();
    3
}

fn main() {
    let f: fn(fn(i32) -> i32) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(bar1), 1);
    let f: fn(unsafe fn()) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(bar2), 2);
    let f: fn(extern "C" fn()) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(bar3), 3);
}
