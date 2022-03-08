// Verify that we can pretty print invalid constants.

#![feature(adt_const_params)]
#![feature(inline_const)]
#![allow(incomplete_features)]

#[derive(Copy, Clone)]
#[repr(u32)]
enum E { A, B, C }

#[derive(Copy, Clone)]
enum Empty {}

// EMIT_MIR invalid_constant.main.ConstProp.diff
fn main() {
    // An invalid char.
    union InvalidChar {
        int: u32,
        chr: char,
    }
    let _invalid_char = const { InvalidChar { int: 0x110001 } };

    // An enum with an invalid tag. Regression test for #93688.
    union InvalidTag {
        int: u32,
        e: E,
    }
    let _invalid_tag = [InvalidTag { int: 4 }];

    // An enum without variants. Regression test for #94073.
    union NoVariants {
        int: u32,
        empty: Empty,
    }
    let _enum_without_variants = [NoVariants { int: 0 }];

    // A non-UTF-8 string slice. Regression test for #75763 and #78520.
    struct Str<const S: &'static str>;
    let _non_utf8_str: Str::<{
        unsafe { std::mem::transmute::<&[u8], &str>(&[0xC0, 0xC1, 0xF5]) }
    }>;
}
