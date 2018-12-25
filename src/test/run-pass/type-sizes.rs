#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![feature(never_type)]

use std::mem::size_of;

struct t {a: u8, b: i8}
struct u {a: u8, b: i8, c: u8}
struct v {a: u8, b: i8, c: v2, d: u32}
struct v2 {u: char, v: u8}
struct w {a: isize, b: ()}
struct x {a: isize, b: (), c: ()}
struct y {x: isize}

enum e1 {
    a(u8, u32), b(u32), c
}
enum e2 {
    a(u32), b
}

#[repr(C, u8)]
enum e3 {
    a([u16; 0], u8), b
}

struct ReorderedStruct {
    a: u8,
    b: u16,
    c: u8
}

enum ReorderedEnum {
    A(u8, u16, u8),
    B(u8, u16, u8),
}

enum EnumEmpty {}

enum EnumSingle1 {
    A,
}

enum EnumSingle2 {
    A = 42 as isize,
}

enum EnumSingle3 {
    A,
    B(!),
}

#[repr(u8)]
enum EnumSingle4 {
    A,
}

#[repr(u8)]
enum EnumSingle5 {
    A = 42 as u8,
}

enum EnumWithMaybeUninhabitedVariant<T> {
    A(&'static ()),
    B(&'static (), T),
    C,
}

enum NicheFilledEnumWithAbsentVariant {
    A(&'static ()),
    B((), !),
    C,
}

pub fn main() {
    assert_eq!(size_of::<u8>(), 1 as usize);
    assert_eq!(size_of::<u32>(), 4 as usize);
    assert_eq!(size_of::<char>(), 4 as usize);
    assert_eq!(size_of::<i8>(), 1 as usize);
    assert_eq!(size_of::<i32>(), 4 as usize);
    assert_eq!(size_of::<t>(), 2 as usize);
    assert_eq!(size_of::<u>(), 3 as usize);
    // Alignment causes padding before the char and the u32.

    assert_eq!(size_of::<v>(),
                16 as usize);
    assert_eq!(size_of::<isize>(), size_of::<usize>());
    assert_eq!(size_of::<w>(), size_of::<isize>());
    assert_eq!(size_of::<x>(), size_of::<isize>());
    assert_eq!(size_of::<isize>(), size_of::<y>());

    // Make sure enum types are the appropriate size, mostly
    // around ensuring alignment is handled properly

    assert_eq!(size_of::<e1>(), 8 as usize);
    assert_eq!(size_of::<e2>(), 8 as usize);
    assert_eq!(size_of::<e3>(), 4 as usize);
    assert_eq!(size_of::<ReorderedStruct>(), 4);
    assert_eq!(size_of::<ReorderedEnum>(), 6);

    assert_eq!(size_of::<EnumEmpty>(), 0);
    assert_eq!(size_of::<EnumSingle1>(), 0);
    assert_eq!(size_of::<EnumSingle2>(), 0);
    assert_eq!(size_of::<EnumSingle3>(), 0);
    assert_eq!(size_of::<EnumSingle4>(), 1);
    assert_eq!(size_of::<EnumSingle5>(), 1);

    assert_eq!(size_of::<EnumWithMaybeUninhabitedVariant<!>>(),
               size_of::<EnumWithMaybeUninhabitedVariant<()>>());
    assert_eq!(size_of::<NicheFilledEnumWithAbsentVariant>(), size_of::<&'static ()>());

    assert_eq!(size_of::<Option<Option<(bool, &())>>>(), size_of::<(bool, &())>());
    assert_eq!(size_of::<Option<Option<(&(), bool)>>>(), size_of::<(bool, &())>());
}
