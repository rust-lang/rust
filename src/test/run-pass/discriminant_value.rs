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

enum ADT {
    First(u32, u32),
    Second(u64)
}

enum NullablePointer {
    Something(&'static u32),
    Nothing
}

static CONST : u32 = 0xBEEF;

pub fn main() {
    unsafe {

        assert_eq!(discriminant_value(&CLike1::A), 0);
        assert_eq!(discriminant_value(&CLike1::B), 1);
        assert_eq!(discriminant_value(&CLike1::C), 2);
        assert_eq!(discriminant_value(&CLike1::D), 3);

        assert_eq!(discriminant_value(&CLike2::A), 5);
        assert_eq!(discriminant_value(&CLike2::B), 2);
        assert_eq!(discriminant_value(&CLike2::C), 19);
        assert_eq!(discriminant_value(&CLike2::D), 20);

        assert_eq!(discriminant_value(&CLike3::A), 5);
        assert_eq!(discriminant_value(&CLike3::B), 6);
        assert_eq!(discriminant_value(&CLike3::C), -1_i8 as u64);
        assert_eq!(discriminant_value(&CLike3::D), 0);

        assert_eq!(discriminant_value(&ADT::First(0,0)), 0);
        assert_eq!(discriminant_value(&ADT::Second(5)), 1);

        assert_eq!(discriminant_value(&NullablePointer::Nothing), 1);
        assert_eq!(discriminant_value(&NullablePointer::Something(&CONST)), 0);

        assert_eq!(discriminant_value(&10), 0);
        assert_eq!(discriminant_value(&"test"), 0);
    }
}
