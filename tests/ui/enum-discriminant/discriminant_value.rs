//@ run-pass
#![allow(stable_features)]
#![feature(core, core_intrinsics)]

extern crate core;
use core::intrinsics::discriminant_value;

enum CLike1 {
    A,
    B,
    C,
    D
}

enum CLike2 {
    A = 5,
    B = 2,
    C = 19,
    D
}

#[repr(i8)]
enum CLike3 {
    A = 5,
    B,
    C = -1,
    D
}

#[allow(dead_code)]
enum ADT {
    First(u32, u32),
    Second(u64)
}

enum NullablePointer {
    Something(#[allow(dead_code)] &'static u32),
    Nothing
}

static CONST : u32 = 0xBEEF;

#[allow(dead_code)]
#[repr(isize)]
enum Mixed {
    Unit = 3,
    Tuple(u16) = 2,
    Struct {
        a: u8,
        b: u16,
    } = 1,
}

pub fn main() {
    assert_eq!(discriminant_value(&CLike1::A), 0isize);
    assert_eq!(discriminant_value(&CLike1::B), 1);
    assert_eq!(discriminant_value(&CLike1::C), 2);
    assert_eq!(discriminant_value(&CLike1::D), 3);

    assert_eq!(discriminant_value(&CLike2::A), 5isize);
    assert_eq!(discriminant_value(&CLike2::B), 2);
    assert_eq!(discriminant_value(&CLike2::C), 19);
    assert_eq!(discriminant_value(&CLike2::D), 20);

    assert_eq!(discriminant_value(&CLike3::A), 5i8);
    assert_eq!(discriminant_value(&CLike3::B), 6);
    assert_eq!(discriminant_value(&CLike3::C), -1);
    assert_eq!(discriminant_value(&CLike3::D), 0);

    assert_eq!(discriminant_value(&ADT::First(0,0)), 0isize);
    assert_eq!(discriminant_value(&ADT::Second(5)), 1);

    assert_eq!(discriminant_value(&NullablePointer::Nothing), 1isize);
    assert_eq!(discriminant_value(&NullablePointer::Something(&CONST)), 0);

    assert_eq!(discriminant_value(&10), 0u8);
    assert_eq!(discriminant_value(&"test"), 0u8);

    assert_eq!(discriminant_value(&Mixed::Unit), 3isize);
    assert_eq!(discriminant_value(&Mixed::Tuple(5)), 2);
    assert_eq!(discriminant_value(&Mixed::Struct{a: 7, b: 11}), 1);
}
