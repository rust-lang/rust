// Verifies that functions with types with const generics as argument types can
// be called through function pointers.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(adt_const_params)]
#![feature(unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct Struct1 {
    x: u16,
    y: u16,
}

#[derive(PartialEq, Eq, ConstParamTy)]
enum Enum1 {
    Variant1,
    Variant2(u8),
}

struct Struct2<const N: usize>([i32; N]);

struct Struct3<const B: bool>(bool);

struct Struct4<const I: i32>(i32);

struct Struct5<const C: char>(char);

struct Struct6<const S: &'static str>(&'static str);

struct Struct7<const S: Struct1>(Struct1);

struct Struct8<const E: Enum1>(Enum1);

struct Struct9<const A: [u16; 2]>([u16; 2]);

struct Struct10<const T: (u16, bool)>((u16, bool));

fn foo1(x: Struct2<2>) -> i32 {
    assert_eq!(x.0, [1, 2]);
    1
}

fn foo2(x: &Struct2<4>) -> i32 {
    assert_eq!(x.0, [1, 2, 3, 4]);
    2
}

fn foo3(x: Struct3<true>) -> i32 {
    assert!(x.0);
    3
}

fn foo4(x: Struct4<-1>) -> i32 {
    assert_eq!(x.0, -1);
    4
}

fn foo5(x: Struct5<'x'>) -> i32 {
    assert_eq!(x.0, 'x');
    5
}

fn foo6(x: Struct6<"hello">) -> i32 {
    assert_eq!(x.0, "hello");
    6
}

fn foo7(x: Struct7<{ Struct1 { x: 1, y: 2 } }>) -> i32 {
    assert_eq!(x.0.x, 1);
    assert_eq!(x.0.y, 2);
    7
}

fn foo8(x: Struct8<{ Enum1::Variant1 }>) -> i32 {
    assert!(matches!(x.0, Enum1::Variant1));
    8
}

fn foo9(x: Struct8<{ Enum1::Variant2(5) }>) -> i32 {
    match x.0 {
        Enum1::Variant1 => unreachable!(),
        Enum1::Variant2(v) => assert_eq!(v, 5),
    }
    9
}

fn foo10(x: Struct9<{ [3, 4] }>) -> i32 {
    assert_eq!(x.0, [3, 4]);
    10
}

fn foo11(x: Struct10<{ (6, true) }>) -> i32 {
    assert_eq!(x.0, (6, true));
    11
}

fn main() {
    let f: fn(Struct2<2>) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(Struct2([1, 2])), 1);
    let f: fn(&Struct2<4>) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(&Struct2([1, 2, 3, 4])), 2);
    let f: fn(Struct3<true>) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(Struct3(true)), 3);
    let f: fn(Struct4<-1>) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f(Struct4(-1)), 4);
    let f: fn(Struct5<'x'>) -> i32 = std::hint::black_box(foo5);
    assert_eq!(f(Struct5('x')), 5);
    let f: fn(Struct6<"hello">) -> i32 = std::hint::black_box(foo6);
    assert_eq!(f(Struct6("hello")), 6);
    let f: fn(Struct7<{ Struct1 { x: 1, y: 2 } }>) -> i32 = std::hint::black_box(foo7);
    assert_eq!(f(Struct7(Struct1 { x: 1, y: 2 })), 7);
    let f: fn(Struct8<{ Enum1::Variant1 }>) -> i32 = std::hint::black_box(foo8);
    assert_eq!(f(Struct8(Enum1::Variant1)), 8);
    let f: fn(Struct8<{ Enum1::Variant2(5) }>) -> i32 = std::hint::black_box(foo9);
    assert_eq!(f(Struct8(Enum1::Variant2(5))), 9);
    let f: fn(Struct9<{ [3, 4] }>) -> i32 = std::hint::black_box(foo10);
    assert_eq!(f(Struct9([3, 4])), 10);
    let f: fn(Struct10<{ (6, true) }>) -> i32 = std::hint::black_box(foo11);
    assert_eq!(f(Struct10((6, true))), 11);
}
