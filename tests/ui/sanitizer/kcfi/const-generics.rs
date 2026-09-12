// Verifies that functions with types with const generics as argument types can
// be called through function pointers.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(adt_const_params)]
#![feature(unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct Struct2 {
    x: u16,
    y: u16,
}

#[derive(PartialEq, Eq, ConstParamTy)]
enum Enum1 {
    Variant1,
    Variant2(u8),
}

struct Struct1<const N: usize>([i32; N]);
struct BoolHolder<const B: bool>(bool);
struct IntHolder<const I: i32>(i32);
struct CharHolder<const C: char>(char);
struct StrHolder<const S: &'static str>(&'static str);
struct StructHolder<const S: Struct2>(Struct2);
struct EnumHolder<const E: Enum1>(Enum1);
struct ArrayHolder<const A: [u16; 2]>([u16; 2]);
struct TupleHolder<const T: (u16, bool)>((u16, bool));

fn foo1(x: Struct1<2>) {
    assert_eq!(x.0, [1, 2]);
}

fn foo2(x: &Struct1<4>) {
    assert_eq!(x.0, [1, 2, 3, 4]);
}

fn foo3(x: BoolHolder<true>) {
    assert!(x.0);
}

fn foo4(x: IntHolder<-1>) {
    assert_eq!(x.0, -1);
}

fn foo5(x: CharHolder<'x'>) {
    assert_eq!(x.0, 'x');
}

fn foo6(x: StrHolder<"hello">) {
    assert_eq!(x.0, "hello");
}

fn foo7(x: StructHolder<{ Struct2 { x: 1, y: 2 } }>) {
    assert_eq!(x.0.x, 1);
    assert_eq!(x.0.y, 2);
}

fn foo8(x: EnumHolder<{ Enum1::Variant1 }>) {
    assert!(matches!(x.0, Enum1::Variant1));
}

fn foo9(x: EnumHolder<{ Enum1::Variant2(5) }>) {
    match x.0 {
        Enum1::Variant1 => unreachable!(),
        Enum1::Variant2(v) => assert_eq!(v, 5),
    }
}

fn foo10(x: ArrayHolder<{ [3, 4] }>) {
    assert_eq!(x.0, [3, 4]);
}

fn foo11(x: TupleHolder<{ (6, true) }>) {
    assert_eq!(x.0, (6, true));
}

fn main() {
    let f: fn(Struct1<2>) = foo1;
    f(Struct1([1, 2]));
    let f: fn(&Struct1<4>) = foo2;
    f(&Struct1([1, 2, 3, 4]));
    let f: fn(BoolHolder<true>) = foo3;
    f(BoolHolder(true));
    let f: fn(IntHolder<-1>) = foo4;
    f(IntHolder(-1));
    let f: fn(CharHolder<'x'>) = foo5;
    f(CharHolder('x'));
    let f: fn(StrHolder<"hello">) = foo6;
    f(StrHolder("hello"));
    let f: fn(StructHolder<{ Struct2 { x: 1, y: 2 } }>) = foo7;
    f(StructHolder(Struct2 { x: 1, y: 2 }));
    let f: fn(EnumHolder<{ Enum1::Variant1 }>) = foo8;
    f(EnumHolder(Enum1::Variant1));
    let f: fn(EnumHolder<{ Enum1::Variant2(5) }>) = foo9;
    f(EnumHolder(Enum1::Variant2(5)));
    let f: fn(ArrayHolder<{ [3, 4] }>) = foo10;
    f(ArrayHolder([3, 4]));
    let f: fn(TupleHolder<{ (6, true) }>) = foo11;
    f(TupleHolder((6, true)));
}
