// Test various non-exhaustive matches for `X..`, `..=X` and `..X` ranges.

#![feature(half_open_range_patterns)]
#![feature(exclusive_range_pattern)]
#![allow(illegal_floating_point_literal_pattern)]

fn main() {}

macro_rules! m {
    ($s:expr, $($t:tt)+) => {
        match $s { $($t)+ => {} }
    }
}

fn floats() {
    m!(0f32, f32::NEG_INFINITY..); //~ ERROR non-exhaustive patterns: `_` not covered
    m!(0f32, ..f32::INFINITY); //~ ERROR non-exhaustive patterns: `_` not covered
}

fn khar() {
    const ALMOST_MAX: char = '\u{10fffe}';
    const ALMOST_MIN: char = '\u{1}';
    const VAL: char = 'a';
    const VAL_1: char = 'b';
    const VAL_2: char = 'c';
    m!('a', ..core::char::MAX); //~ ERROR non-exhaustive patterns
    m!('a', ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
    m!('a', ALMOST_MIN..); //~ ERROR non-exhaustive patterns
    m!('a', ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
    m!('a', ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
    m!('a', ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
}

mod unsigned {
    fn u8() {
        const ALMOST_MAX: u8 = u8::MAX - 1;
        const ALMOST_MIN: u8 = u8::MIN + 1;
        const VAL: u8 = 42;
        const VAL_1: u8 = VAL + 1;
        const VAL_2: u8 = VAL + 2;
        m!(0, ..u8::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn u16() {
        const ALMOST_MAX: u16 = u16::MAX - 1;
        const ALMOST_MIN: u16 = u16::MIN + 1;
        const VAL: u16 = 42;
        const VAL_1: u16 = VAL + 1;
        const VAL_2: u16 = VAL + 2;
        m!(0, ..u16::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn u32() {
        const ALMOST_MAX: u32 = u32::MAX - 1;
        const ALMOST_MIN: u32 = u32::MIN + 1;
        const VAL: u32 = 42;
        const VAL_1: u32 = VAL + 1;
        const VAL_2: u32 = VAL + 2;
        m!(0, ..u32::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn u64() {
        const ALMOST_MAX: u64 = u64::MAX - 1;
        const ALMOST_MIN: u64 = u64::MIN + 1;
        const VAL: u64 = 42;
        const VAL_1: u64 = VAL + 1;
        const VAL_2: u64 = VAL + 2;
        m!(0, ..u64::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn u128() {
        const ALMOST_MAX: u128 = u128::MAX - 1;
        const ALMOST_MIN: u128 = u128::MIN + 1;
        const VAL: u128 = 42;
        const VAL_1: u128 = VAL + 1;
        const VAL_2: u128 = VAL + 2;
        m!(0, ..u128::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
}

mod signed {
    fn i8() {
        const ALMOST_MAX: i8 = i8::MAX - 1;
        const ALMOST_MIN: i8 = i8::MIN + 1;
        const VAL: i8 = 42;
        const VAL_1: i8 = VAL + 1;
        const VAL_2: i8 = VAL + 2;
        m!(0, ..i8::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn i16() {
        const ALMOST_MAX: i16 = i16::MAX - 1;
        const ALMOST_MIN: i16 = i16::MIN + 1;
        const VAL: i16 = 42;
        const VAL_1: i16 = VAL + 1;
        const VAL_2: i16 = VAL + 2;
        m!(0, ..i16::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn i32() {
        const ALMOST_MAX: i32 = i32::MAX - 1;
        const ALMOST_MIN: i32 = i32::MIN + 1;
        const VAL: i32 = 42;
        const VAL_1: i32 = VAL + 1;
        const VAL_2: i32 = VAL + 2;
        m!(0, ..i32::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn i64() {
        const ALMOST_MAX: i64 = i64::MAX - 1;
        const ALMOST_MIN: i64 = i64::MIN + 1;
        const VAL: i64 = 42;
        const VAL_1: i64 = VAL + 1;
        const VAL_2: i64 = VAL + 2;
        m!(0, ..i64::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
    fn i128() {
        const ALMOST_MAX: i128 = i128::MAX - 1;
        const ALMOST_MIN: i128 = i128::MIN + 1;
        const VAL: i128 = 42;
        const VAL_1: i128 = VAL + 1;
        const VAL_2: i128 = VAL + 2;
        m!(0, ..i128::MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ALMOST_MIN..); //~ ERROR non-exhaustive patterns
        m!(0, ..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
        m!(0, ..=VAL | VAL_2..); //~ ERROR non-exhaustive patterns
        m!(0, ..VAL_1 | VAL_2..); //~ ERROR non-exhaustive patterns
    }
}
