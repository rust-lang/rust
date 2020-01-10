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
    test_int!(0u8, core::u8::MIN, core::u8::MAX);
    test_int!(0u16, core::u16::MIN, core::u16::MAX);
    test_int!(0u32, core::u32::MIN, core::u32::MAX);
    test_int!(0u64, core::u64::MIN, core::u64::MAX);
    test_int!(0u128, core::u128::MIN, core::u128::MAX);
}

fn signed_int() {
    test_int!(0i8, core::i8::MIN, core::i8::MAX);
    test_int!(0i16, core::i16::MIN, core::i16::MAX);
    test_int!(0i32, core::i32::MIN, core::i32::MAX);
    test_int!(0i64, core::i64::MIN, core::i64::MAX);
    test_int!(0i128, core::i128::MIN, core::i128::MAX);
}

fn khar() {
    m!('a', ..=core::char::MAX);
    m!('a', '\u{0}'..);
    m!('a', ..='\u{D7FF}' | '\u{E000}'..);
    m!('a', ..'\u{D7FF}' | '\u{D7FF}' | '\u{E000}'..);
}
