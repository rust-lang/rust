//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::mem;

#[repr(packed)]
struct P1S4(u8,[u8;  3]);

#[repr(packed(2))]
struct P2S4(u8,[u8;  3]);

#[repr(packed)]
struct P1S5(u8, u32);

#[repr(packed(2))]
struct P2S6(u8, u32);

#[repr(packed)]
struct P1S13(i64, f32, u8);

#[repr(packed(2))]
struct P2S14(i64, f32, u8);

#[repr(packed(4))]
struct P4S16(u8, f32, i64, u16);

#[repr(C, packed(4))]
struct P4CS20(u8, f32, i64, u16);

enum Foo {
    Bar = 1,
    Baz = 2
}

#[repr(packed)]
struct P1S3_Foo(u8, u16, Foo);

#[repr(packed(2))]
struct P2_Foo(Foo);

#[repr(packed(2))]
struct P2S3_Foo(u8, u16, Foo);

#[repr(packed)]
struct P1S7_Option(f32, u8, u16, Option<Box<f64>>);

#[repr(packed(2))]
struct P2_Option(Option<Box<f64>>);

#[repr(packed(2))]
struct P2S7_Option(f32, u8, u16, Option<Box<f64>>);

fn align_to(value: usize, align: usize) -> usize {
    (value + (align - 1)) & !(align - 1)
}

macro_rules! check {
    ($t:ty, $align:expr, $size:expr) => ({
        assert_eq!(mem::align_of::<$t>(), $align);
        assert_eq!(mem::size_of::<$t>(), $size);
    });
}

pub fn main() {
    check!(P1S4, 1, 4);
    check!(P1S5, 1, 5);
    check!(P1S13, 1, 13);
    check!(P1S3_Foo, 1, 3 + mem::size_of::<Foo>());
    check!(P1S7_Option, 1, 7 + mem::size_of::<Option<Box<f64>>>());

    check!(P2S4, 1, 4);
    check!(P2S6, 2, 6);
    check!(P2S14, 2, 14);
    check!(P4S16, 4, 16);
    check!(P4CS20, 4, 20);
    check!(P2S3_Foo, 2, align_to(3 + mem::size_of::<P2_Foo>(), 2));
    check!(P2S7_Option, 2, align_to(7 + mem::size_of::<P2_Option>(), 2));
}
