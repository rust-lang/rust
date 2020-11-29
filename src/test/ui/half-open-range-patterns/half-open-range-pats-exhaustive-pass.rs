// check-pass

// Test various exhaustive matches for `X..`, `..=X` and `..X` ranges.

#![feature(half_open_range_patterns)]
#![feature(exclusive_range_pattern)]

fn main() {}

macro_rules! m {
    ($s:expr, $($t:tt)+) => {
        match $s { $($t)+ => {} }
    }
}

macro_rules! test_int {
    ($s:expr, $min:path, $max:path) => {
        m!($s, $min..);
        m!($s, $min..5 | 5..);
        m!($s, ..5 | 5..);
        m!($s, ..=4 | 5..);
        m!($s, ..=$max);
        m!($s, ..$max | $max);
        m!(($s, true), (..5, true) | (5.., true) | ($min.., false));
    }
}

fn unsigned_int() {
    test_int!(0u8, u8::MIN, u8::MAX);
    test_int!(0u16, u16::MIN, u16::MAX);
    test_int!(0u32, u32::MIN, u32::MAX);
    test_int!(0u64, u64::MIN, u64::MAX);
    test_int!(0u128, u128::MIN, u128::MAX);
}

fn signed_int() {
    test_int!(0i8, i8::MIN, i8::MAX);
    test_int!(0i16, i16::MIN, i16::MAX);
    test_int!(0i32, i32::MIN, i32::MAX);
    test_int!(0i64, i64::MIN, i64::MAX);
    test_int!(0i128, i128::MIN, i128::MAX);
}

fn khar() {
    m!('a', ..=core::char::MAX);
    m!('a', '\u{0}'..);
    m!('a', ..='\u{D7FF}' | '\u{E000}'..);
    m!('a', ..'\u{D7FF}' | '\u{D7FF}' | '\u{E000}'..);
}
