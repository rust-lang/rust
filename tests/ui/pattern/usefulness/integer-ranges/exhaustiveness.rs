#![allow(overlapping_range_endpoints)]
#![allow(non_contiguous_range_endpoints)]
#![deny(unreachable_patterns)]

macro_rules! m {
    ($s:expr, $($t:tt)+) => {
        match $s { $($t)+ => {} }
    }
}

macro_rules! test_int {
    ($s:expr, $min:path, $max:path) => {
        m!($s, $min..=$max);
        m!($s, $min..5 | 5..=$max);
        m!($s, $min..=4 | 5..=$max);
        m!($s, $min..$max | $max);
        m!(($s, true), ($min..5, true) | (5..=$max, true) | ($min..=$max, false));
    }
}

fn main() {
    test_int!(0u8, u8::MIN, u8::MAX);
    test_int!(0u16, u16::MIN, u16::MAX);
    test_int!(0u32, u32::MIN, u32::MAX);
    test_int!(0u64, u64::MIN, u64::MAX);
    test_int!(0u128, u128::MIN, u128::MAX);

    test_int!(0i8, i8::MIN, i8::MAX);
    test_int!(0i16, i16::MIN, i16::MAX);
    test_int!(0i32, i32::MIN, i32::MAX);
    test_int!(0i64, i64::MIN, i64::MAX);
    test_int!(0i128, i128::MIN, i128::MAX);

    m!('a', '\u{0}'..=char::MAX);
    m!('a', '\u{0}'..='\u{10_FFFF}');
    // We can get away with just covering the following two ranges, which correspond to all valid
    // Unicode Scalar Values.
    m!('a', '\u{0}'..='\u{D7FF}' | '\u{E000}'..=char::MAX);
    m!('a', '\u{0}'..'\u{D7FF}' | '\u{D7FF}' | '\u{E000}'..=char::MAX);

    let 0..=255 = 0u8;
    let -128..=127 = 0i8;
    let -2147483648..=2147483647 = 0i32;
    let '\u{0000}'..='\u{10FFFF}' = 'v';

    // Almost exhaustive
    m!(0u8, 0..255); //~ ERROR non-exhaustive patterns
    m!(0u8, 0..=254); //~ ERROR non-exhaustive patterns
    m!(0u8, 1..=255); //~ ERROR non-exhaustive patterns
    m!(0u8, 0..42 | 43..=255); //~ ERROR non-exhaustive patterns
    m!(0i8, -128..127); //~ ERROR non-exhaustive patterns
    m!(0i8, -128..=126); //~ ERROR non-exhaustive patterns
    m!(0i8, -127..=127); //~ ERROR non-exhaustive patterns
    match 0i8 { //~ ERROR non-exhaustive patterns
        i8::MIN ..= -1 => {}
        1 ..= i8::MAX => {}
    }
    const ALMOST_MAX: u128 = u128::MAX - 1;
    m!(0u128, 0..=ALMOST_MAX); //~ ERROR non-exhaustive patterns
    m!(0u128, 0..=4); //~ ERROR non-exhaustive patterns
    m!(0u128, 1..=u128::MAX); //~ ERROR non-exhaustive patterns

    // More complicatedly (non-)exhaustive
    match 0u8 {
        0 ..= 30 => {}
        20 ..= 70 => {}
        50 ..= 255 => {}
    }
    match (0u8, true) { //~ ERROR non-exhaustive patterns
        (0 ..= 125, false) => {}
        (128 ..= 255, false) => {}
        (0 ..= 255, true) => {}
    }
    match (0u8, true) { // ok
        (0 ..= 125, false) => {}
        (128 ..= 255, false) => {}
        (0 ..= 255, true) => {}
        (125 .. 128, false) => {}
    }
    match (true, 0u8) {
        (true, 0 ..= 255) => {}
        (false, 0 ..= 125) => {}
        (false, 128 ..= 255) => {}
        (false, 125 .. 128) => {}
    }
    match Some(0u8) {
        None => {}
        Some(0 ..= 125) => {}
        Some(128 ..= 255) => {}
        Some(125 .. 128) => {}
    }
    const FOO: u8 = 41;
    const BAR: &u8 = &42;
    match &0u8 {
        0..41 => {}
        &FOO => {}
        BAR => {}
        43..=255 => {}
    }

}
