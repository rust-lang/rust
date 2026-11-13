// Verifies that functions with user-defined types (i.e., structs, enums,
// unions, and extern types) as argument types can be called through function
// pointers.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(extern_types)]

struct Struct1(i32);

enum Enum1 {
    Variant1(i32),
}

union Union1 {
    f: i32,
}

#[repr(C)]
struct Struct2 {
    f: i32,
}

struct Struct3<T>(T);

unsafe extern "C" {
    type Type1;
}

fn foo1(x: Struct1) -> i32 {
    assert_eq!(x.0, 1);
    1
}

fn foo2(x: Enum1) -> i32 {
    let Enum1::Variant1(y) = x;
    assert_eq!(y, 2);
    2
}

fn foo3(x: Union1) -> i32 {
    assert_eq!(unsafe { x.f }, 3);
    3
}

fn foo4(x: Struct2) -> i32 {
    assert_eq!(x.f, 4);
    4
}

fn foo5(x: Struct3<i32>) -> i32 {
    assert_eq!(x.0, 5);
    5
}

fn foo6(_: *const Type1) -> i32 {
    6
}

extern "C" fn foo7(x: Struct2) -> i32 {
    assert_eq!(x.f, 7);
    7
}

fn main() {
    let f: fn(Struct1) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(Struct1(1)), 1);
    let f: fn(Enum1) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(Enum1::Variant1(2)), 2);
    let f: fn(Union1) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(Union1 { f: 3 }), 3);
    let f: fn(Struct2) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f(Struct2 { f: 4 }), 4);
    let f: fn(Struct3<i32>) -> i32 = std::hint::black_box(foo5);
    assert_eq!(f(Struct3(5)), 5);
    let f: fn(*const Type1) -> i32 = std::hint::black_box(foo6);
    assert_eq!(f(&() as *const () as *const Type1), 6);
    let f: extern "C" fn(Struct2) -> i32 = std::hint::black_box(foo7);
    assert_eq!(f(Struct2 { f: 7 }), 7);
}
