// Verifies that functions with primitive types as argument and return types can
// be called through function pointers.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(f128)]
#![feature(f16)]
fn foo1(_: ()) -> i32 {
    1
}
fn foo2(_: bool) -> i32 {
    2
}
fn foo3(_: char) -> i32 {
    3
}
fn foo4(_: f32) -> i32 {
    4
}
fn foo5(_: f64) -> i32 {
    5
}
fn foo6(_: i8) -> i32 {
    6
}
fn foo7(_: i16) -> i32 {
    7
}
fn foo8(_: i32) -> i32 {
    8
}
fn foo9(_: i64) -> i32 {
    9
}
fn foo10(_: i128) -> i32 {
    10
}
fn foo11(_: isize) -> i32 {
    11
}
fn foo12(_: u8) -> i32 {
    12
}
fn foo13(_: u16) -> i32 {
    13
}
fn foo14(_: u32) -> i32 {
    14
}
fn foo15(_: u64) -> i32 {
    15
}
fn foo16(_: u128) -> i32 {
    16
}
fn foo17(_: usize) -> i32 {
    17
}
fn foo18(_: f16) -> i32 {
    18
}
fn foo19(_: f128) -> i32 {
    19
}
fn foo20() -> ! {
    std::process::exit(0)
}

fn main() {
    let f: fn(()) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(()), 1);
    let f: fn(bool) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(false), 2);
    let f: fn(char) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f('a'), 3);
    let f: fn(f32) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f(0f32), 4);
    let f: fn(f64) -> i32 = std::hint::black_box(foo5);
    assert_eq!(f(0f64), 5);
    let f: fn(i8) -> i32 = std::hint::black_box(foo6);
    assert_eq!(f(0i8), 6);
    let f: fn(i16) -> i32 = std::hint::black_box(foo7);
    assert_eq!(f(0i16), 7);
    let f: fn(i32) -> i32 = std::hint::black_box(foo8);
    assert_eq!(f(0i32), 8);
    let f: fn(i64) -> i32 = std::hint::black_box(foo9);
    assert_eq!(f(0i64), 9);
    let f: fn(i128) -> i32 = std::hint::black_box(foo10);
    assert_eq!(f(0i128), 10);
    let f: fn(isize) -> i32 = std::hint::black_box(foo11);
    assert_eq!(f(0isize), 11);
    let f: fn(u8) -> i32 = std::hint::black_box(foo12);
    assert_eq!(f(0u8), 12);
    let f: fn(u16) -> i32 = std::hint::black_box(foo13);
    assert_eq!(f(0u16), 13);
    let f: fn(u32) -> i32 = std::hint::black_box(foo14);
    assert_eq!(f(0u32), 14);
    let f: fn(u64) -> i32 = std::hint::black_box(foo15);
    assert_eq!(f(0u64), 15);
    let f: fn(u128) -> i32 = std::hint::black_box(foo16);
    assert_eq!(f(0u128), 16);
    let f: fn(usize) -> i32 = std::hint::black_box(foo17);
    assert_eq!(f(0usize), 17);
    let f: fn(f16) -> i32 = std::hint::black_box(foo18);
    assert_eq!(f(0f16), 18);
    let f: fn(f128) -> i32 = std::hint::black_box(foo19);
    assert_eq!(f(0f128), 19);
    // The never type can only be returned, so this must be called last
    let f: fn() -> ! = std::hint::black_box(foo20);
    f();
}
