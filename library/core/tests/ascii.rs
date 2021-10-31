use core::char::from_u32;

#[test]
fn test_is_ascii() {
    assert!(b"".is_ascii());
    assert!(b"banana\0\x7F".is_ascii());
    assert!(b"banana\0\x7F".iter().all(|b| b.is_ascii()));
    assert!(!b"Vi\xe1\xbb\x87t Nam".is_ascii());
    assert!(!b"Vi\xe1\xbb\x87t Nam".iter().all(|b| b.is_ascii()));
    assert!(!b"\xe1\xbb\x87".iter().any(|b| b.is_ascii()));

    assert!("".is_ascii());
    assert!("banana\0\u{7F}".is_ascii());
    assert!("banana\0\u{7F}".chars().all(|c| c.is_ascii()));
    assert!(!"ประเทศไทย中华Việt Nam".chars().all(|c| c.is_ascii()));
    assert!(!"ประเทศไทย中华ệ ".chars().any(|c| c.is_ascii()));
}

#[test]
fn test_to_ascii_uppercase() {
    assert_eq!("url()URL()uRl()ürl".to_ascii_uppercase(), "URL()URL()URL()üRL");
    assert_eq!("hıKß".to_ascii_uppercase(), "HıKß");

    for i in 0..501 {
        let upper =
            if 'a' as u32 <= i && i <= 'z' as u32 { i + 'A' as u32 - 'a' as u32 } else { i };
        assert_eq!(
            (from_u32(i).unwrap()).to_string().to_ascii_uppercase(),
            (from_u32(upper).unwrap()).to_string()
        );
    }
}

#[test]
fn test_to_ascii_lowercase() {
    assert_eq!("url()URL()uRl()Ürl".to_ascii_lowercase(), "url()url()url()Ürl");
    // Dotted capital I, Kelvin sign, Sharp S.
    assert_eq!("HİKß".to_ascii_lowercase(), "hİKß");

    for i in 0..501 {
        let lower =
            if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 } else { i };
        assert_eq!(
            (from_u32(i).unwrap()).to_string().to_ascii_lowercase(),
            (from_u32(lower).unwrap()).to_string()
        );
    }
}

#[test]
fn test_make_ascii_lower_case() {
    macro_rules! test {
        ($from: expr, $to: expr) => {{
            let mut x = $from;
            x.make_ascii_lowercase();
            assert_eq!(x, $to);
        }};
    }
    test!(b'A', b'a');
    test!(b'a', b'a');
    test!(b'!', b'!');
    test!('A', 'a');
    test!('À', 'À');
    test!('a', 'a');
    test!('!', '!');
    test!(b"H\xc3\x89".to_vec(), b"h\xc3\x89");
    test!("HİKß".to_string(), "hİKß");
}

#[test]
fn test_make_ascii_upper_case() {
    macro_rules! test {
        ($from: expr, $to: expr) => {{
            let mut x = $from;
            x.make_ascii_uppercase();
            assert_eq!(x, $to);
        }};
    }
    test!(b'a', b'A');
    test!(b'A', b'A');
    test!(b'!', b'!');
    test!('a', 'A');
    test!('à', 'à');
    test!('A', 'A');
    test!('!', '!');
    test!(b"h\xc3\xa9".to_vec(), b"H\xc3\xa9");
    test!("hıKß".to_string(), "HıKß");

    let mut x = "Hello".to_string();
    x[..3].make_ascii_uppercase(); // Test IndexMut on String.
    assert_eq!(x, "HELlo")
}

#[test]
fn test_eq_ignore_ascii_case() {
    assert!("url()URL()uRl()Ürl".eq_ignore_ascii_case("url()url()url()Ürl"));
    assert!(!"Ürl".eq_ignore_ascii_case("ürl"));
    // Dotted capital I, Kelvin sign, Sharp S.
    assert!("HİKß".eq_ignore_ascii_case("hİKß"));
    assert!(!"İ".eq_ignore_ascii_case("i"));
    assert!(!"K".eq_ignore_ascii_case("k"));
    assert!(!"ß".eq_ignore_ascii_case("s"));

    for i in 0..501 {
        let lower =
            if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 } else { i };
        assert!(
            (from_u32(i).unwrap())
                .to_string()
                .eq_ignore_ascii_case(&from_u32(lower).unwrap().to_string())
        );
    }
}

#[test]
fn inference_works() {
    let x = "a".to_string();
    let _ = x.eq_ignore_ascii_case("A");
}

// Shorthands used by the is_ascii_* tests.
macro_rules! assert_all {
    ($what:ident, $($str:tt),+) => {{
        $(
            for b in $str.chars() {
                if !b.$what() {
                    panic!("expected {}({}) but it isn't",
                           stringify!($what), b);
                }
            }
            for b in $str.as_bytes().iter() {
                if !b.$what() {
                    panic!("expected {}(0x{:02x})) but it isn't",
                           stringify!($what), b);
                }
            }
        )+
    }};
    ($what:ident, $($str:tt),+,) => (assert_all!($what,$($str),+))
}
macro_rules! assert_none {
    ($what:ident, $($str:tt),+) => {{
        $(
            for b in $str.chars() {
                if b.$what() {
                    panic!("expected not-{}({}) but it is",
                           stringify!($what), b);
                }
            }
            for b in $str.as_bytes().iter() {
                if b.$what() {
                    panic!("expected not-{}(0x{:02x})) but it is",
                           stringify!($what), b);
                }
            }
        )+
    }};
    ($what:ident, $($str:tt),+,) => (assert_none!($what,$($str),+))
}

#[test]
fn test_is_ascii_alphabetic() {
    assert_all!(
        is_ascii_alphabetic,
        "",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
    );
    assert_none!(
        is_ascii_alphabetic,
        "0123456789",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_uppercase() {
    assert_all!(is_ascii_uppercase, "", "ABCDEFGHIJKLMNOQPRSTUVWXYZ",);
    assert_none!(
        is_ascii_uppercase,
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_lowercase() {
    assert_all!(is_ascii_lowercase, "abcdefghijklmnopqrstuvwxyz",);
    assert_none!(
        is_ascii_lowercase,
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        "0123456789",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_alphanumeric() {
    assert_all!(
        is_ascii_alphanumeric,
        "",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        "0123456789",
    );
    assert_none!(
        is_ascii_alphanumeric,
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_digit() {
    assert_all!(is_ascii_digit, "", "0123456789",);
    assert_none!(
        is_ascii_digit,
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_hexdigit() {
    assert_all!(is_ascii_hexdigit, "", "0123456789", "abcdefABCDEF",);
    assert_none!(
        is_ascii_hexdigit,
        "ghijklmnopqrstuvwxyz",
        "GHIJKLMNOQPRSTUVWXYZ",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_punctuation() {
    assert_all!(is_ascii_punctuation, "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",);
    assert_none!(
        is_ascii_punctuation,
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        "0123456789",
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_graphic() {
    assert_all!(
        is_ascii_graphic,
        "",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        "0123456789",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    );
    assert_none!(
        is_ascii_graphic,
        " \t\n\x0c\r",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_whitespace() {
    assert_all!(is_ascii_whitespace, "", " \t\n\x0c\r",);
    assert_none!(
        is_ascii_whitespace,
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        "0123456789",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x0b\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
}

#[test]
fn test_is_ascii_control() {
    assert_all!(
        is_ascii_control,
        "",
        "\x00\x01\x02\x03\x04\x05\x06\x07",
        "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
        "\x10\x11\x12\x13\x14\x15\x16\x17",
        "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
        "\x7f",
    );
    assert_none!(
        is_ascii_control,
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        "0123456789",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        " ",
    );
}

// `is_ascii` does a good amount of pointer manipulation and has
// alignment-dependent computation. This is all sanity-checked via
// `debug_assert!`s, so we test various sizes/alignments thoroughly versus an
// "obviously correct" baseline function.
#[test]
fn test_is_ascii_align_size_thoroughly() {
    // The "obviously-correct" baseline mentioned above.
    fn is_ascii_baseline(s: &[u8]) -> bool {
        s.iter().all(|b| b.is_ascii())
    }

    // Helper to repeat `l` copies of `b0` followed by `l` copies of `b1`.
    fn repeat_concat(b0: u8, b1: u8, l: usize) -> Vec<u8> {
        use core::iter::repeat;
        repeat(b0).take(l).chain(repeat(b1).take(l)).collect()
    }

    // Miri is too slow
    let iter = if cfg!(miri) { 0..20 } else { 0..100 };

    for i in iter {
        #[cfg(not(miri))]
        let cases = &[
            b"a".repeat(i),
            b"\0".repeat(i),
            b"\x7f".repeat(i),
            b"\x80".repeat(i),
            b"\xff".repeat(i),
            repeat_concat(b'a', 0x80u8, i),
            repeat_concat(0x80u8, b'a', i),
        ];

        #[cfg(miri)]
        let cases = &[b"a".repeat(i), b"\x80".repeat(i), repeat_concat(b'a', 0x80u8, i)];

        for case in cases {
            for pos in 0..=case.len() {
                // Potentially misaligned head
                let prefix = &case[pos..];
                assert_eq!(is_ascii_baseline(prefix), prefix.is_ascii(),);

                // Potentially misaligned tail
                let suffix = &case[..case.len() - pos];

                assert_eq!(is_ascii_baseline(suffix), suffix.is_ascii(),);

                // Both head and tail are potentially misaligned
                let mid = &case[(pos / 2)..(case.len() - (pos / 2))];
                assert_eq!(is_ascii_baseline(mid), mid.is_ascii(),);
            }
        }
    }
}

#[test]
fn ascii_const() {
    // test that the `is_ascii` methods of `char` and `u8` are usable in a const context

    const CHAR_IS_ASCII: bool = 'a'.is_ascii();
    assert!(CHAR_IS_ASCII);

    const BYTE_IS_ASCII: bool = 97u8.is_ascii();
    assert!(BYTE_IS_ASCII);
}

#[test]
fn ascii_ctype_const() {
    macro_rules! suite {
        ( $( $fn:ident => [$a:ident, $A:ident, $nine:ident, $dot:ident, $space:ident]; )* ) => {
            $(
                mod $fn {
                    const CHAR_A_LOWER: bool = 'a'.$fn();
                    const CHAR_A_UPPER: bool = 'A'.$fn();
                    const CHAR_NINE: bool = '9'.$fn();
                    const CHAR_DOT: bool = '.'.$fn();
                    const CHAR_SPACE: bool = ' '.$fn();

                    const U8_A_LOWER: bool = b'a'.$fn();
                    const U8_A_UPPER: bool = b'A'.$fn();
                    const U8_NINE: bool = b'9'.$fn();
                    const U8_DOT: bool = b'.'.$fn();
                    const U8_SPACE: bool = b' '.$fn();

                    pub fn run() {
                        assert_eq!(CHAR_A_LOWER, $a);
                        assert_eq!(CHAR_A_UPPER, $A);
                        assert_eq!(CHAR_NINE, $nine);
                        assert_eq!(CHAR_DOT, $dot);
                        assert_eq!(CHAR_SPACE, $space);

                        assert_eq!(U8_A_LOWER, $a);
                        assert_eq!(U8_A_UPPER, $A);
                        assert_eq!(U8_NINE, $nine);
                        assert_eq!(U8_DOT, $dot);
                        assert_eq!(U8_SPACE, $space);
                    }
                }
            )*

            $( $fn::run(); )*
        }
    }

    suite! {
        //                        'a'    'A'    '9'    '.'    ' '
        is_ascii_alphabetic   => [true,  true,  false, false, false];
        is_ascii_uppercase    => [false, true,  false, false, false];
        is_ascii_lowercase    => [true,  false, false, false, false];
        is_ascii_alphanumeric => [true,  true,  true,  false, false];
        is_ascii_digit        => [false, false, true,  false, false];
        is_ascii_hexdigit     => [true,  true,  true,  false, false];
        is_ascii_punctuation  => [false, false, false, true,  false];
        is_ascii_graphic      => [true,  true,  true,  true,  false];
        is_ascii_whitespace   => [false, false, false, false, true];
        is_ascii_control      => [false, false, false, false, false];
    }
}
