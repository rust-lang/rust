use core::ascii::Char;
use core::fmt::Write;

/// Tests Display implementation for ascii::Char.
#[test]
fn test_display() {
    let want = (0..128u8).map(|b| b as char).collect::<String>();
    let mut got = String::with_capacity(128);
    for byte in 0..128 {
        write!(&mut got, "{}", Char::from_u8(byte).unwrap()).unwrap();
    }
    assert_eq!(want, got);
}

/// Tests Debug implementation for ascii::Char.
#[test]
fn test_debug_control() {
    for byte in 0..128u8 {
        let mut want = format!("{:?}", byte as char);
        // `char` uses `'\u{#}'` representation where ascii::char uses `'\x##'`.
        // Transform former into the latter.
        if let Some(rest) = want.strip_prefix("'\\u{") {
            want = format!("'\\x{:0>2}'", rest.strip_suffix("}'").unwrap());
        }
        let chr = core::ascii::Char::from_u8(byte).unwrap();
        assert_eq!(want, format!("{chr:?}"), "byte: {byte}");
    }
}

/// Tests Extend implementation for ascii::Char.
#[test]
fn test_extend() {
    let mut s = String::from("abc");
    s.extend_one(Char::SmallD);
    assert_eq!(s, String::from("abcd"));

    let mut s = String::from("abc");
    s.extend(Char::CapitalA..=Char::CapitalC);
    assert_eq!(s, String::from("abcABC"));
}

/// Tests the output of `Char::from_u8`.
#[test]
fn test_from_u8() {
    for n in 0..128_u8 {
        let ch = Char::from_u8(n).unwrap();
        assert_eq!(n, ch as u8);
    }
    for n in 128_u8..=u8::MAX {
        assert!(Char::from_u8(n).is_none());
    }
}

macro_rules! test_from {
    ($($fn:ident : $ty:ident),*) => {
        $(
            /// Tests `$ty::from(ascii_char)`.
            #[test]
            fn $fn() {
                for n in 0..128_u8 {
                    let ch = Char::from_u8(n).unwrap();
                    let result = $ty::from(ch);

                    // Check that `result` is nonnegative.
                    assert!(
                        result >= 0 as $ty,
                        concat!("`", stringify!($ty), "::from(Char::from_u8({}))` is negative ({})."),
                        n,
                        result
                    );
                    // Check for the correct value, assuming that `result` is nonnegative.
                    assert_eq!(n as u128, result as u128);
                }
            }
        )*
    };
}

test_from!(
    test_i8_from: i8,
    test_i16_from: i16,
    test_i32_from: i32,
    test_i64_from: i64,
    test_i128_from: i128,
    test_u8_from: u8,
    test_u16_from: u16,
    test_u32_from: u32,
    test_u64_from: u64,
    test_u128_from: u128,
    test_char_from: char
);
