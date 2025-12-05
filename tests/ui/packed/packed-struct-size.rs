//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::mem;

#[repr(packed)]
struct P1S4 {
    a: u8,
    b: [u8;  3],
}

#[repr(packed(2))]
struct P2S4 {
    a: u8,
    b: [u8;  3],
}

#[repr(packed)]
struct P1S5 {
    a: u8,
    b: u32
}

#[repr(packed(2))]
struct P2S2 {
    a: u8,
    b: u8
}

#[repr(packed(2))]
struct P2S6 {
    a: u8,
    b: u32
}

#[repr(packed(2))]
struct P2S12 {
    a: u32,
    b: u64
}

#[repr(packed)]
struct P1S13 {
    a: i64,
    b: f32,
    c: u8,
}

#[repr(packed(2))]
struct P2S14 {
    a: i64,
    b: f32,
    c: u8,
}

#[repr(packed(4))]
struct P4S16 {
    a: u8,
    b: f32,
    c: i64,
    d: u16,
}

#[repr(C, packed(4))]
struct P4CS20 {
    a: u8,
    b: f32,
    c: i64,
    d: u16,
}

enum Foo {
    Bar = 1,
    Baz = 2
}

#[repr(packed)]
struct P1S3_Foo {
    a: u8,
    b: u16,
    c: Foo
}

#[repr(packed(2))]
struct P2_Foo {
    a: Foo,
}

#[repr(packed(2))]
struct P2S3_Foo {
    a: u8,
    b: u16,
    c: Foo
}

#[repr(packed)]
struct P1S7_Option {
    a: f32,
    b: u8,
    c: u16,
    d: Option<Box<f64>>
}

#[repr(packed(2))]
struct P2_Option {
    a: Option<Box<f64>>
}

#[repr(packed(2))]
struct P2S7_Option {
    a: f32,
    b: u8,
    c: u16,
    d: Option<Box<f64>>
}

// Placing packed structs in statics should work
static TEST_P1S4: P1S4 = P1S4 { a: 1, b: [2, 3, 4] };
static TEST_P1S5: P1S5 = P1S5 { a: 3, b: 67 };
static TEST_P1S3_Foo: P1S3_Foo = P1S3_Foo { a: 1, b: 2, c: Foo::Baz };
static TEST_P2S2: P2S2 = P2S2 { a: 1, b: 2 };
static TEST_P2S4: P2S4 = P2S4 { a: 1, b: [2, 3, 4] };
static TEST_P2S6: P2S6 = P2S6 { a: 1, b: 2 };
static TEST_P2S12: P2S12 = P2S12 { a: 1, b: 2 };
static TEST_P4S16: P4S16 = P4S16 { a: 1, b: 2.0, c: 3, d: 4 };
static TEST_P4CS20: P4CS20 = P4CS20 { a: 1, b: 2.0, c: 3, d: 4 };

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

    check!(P2S2, 1, 2);
    check!(P2S4, 1, 4);
    check!(P2S6, 2, 6);
    check!(P2S12, 2, 12);
    check!(P2S14, 2, 14);
    check!(P4S16, 4, 16);
    check!(P4CS20, 4, 20);
    check!(P2S3_Foo, 2, align_to(3 + mem::size_of::<P2_Foo>(), 2));
    check!(P2S7_Option, 2, align_to(7 + mem::size_of::<P2_Option>(), 2));
}
