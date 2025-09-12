use std::char::MAX_LEN_UTF8;
use std::str::FromStr;
use std::{char, str};

#[test]
fn test_convert() {
    assert_eq!(u32::from('a'), 0x61);
    assert_eq!(u64::from('b'), 0x62);
    assert_eq!(u128::from('c'), 0x63);
    assert_eq!(char::from(b'\0'), '\0');
    assert_eq!(char::from(b'a'), 'a');
    assert_eq!(char::from(b'\xFF'), '\u{FF}');
    assert_eq!(char::try_from(0_u32), Ok('\0'));
    assert_eq!(char::try_from(0x61_u32), Ok('a'));
    assert_eq!(char::try_from(0xD7FF_u32), Ok('\u{D7FF}'));
    assert!(char::try_from(0xD800_u32).is_err());
    assert!(char::try_from(0xDFFF_u32).is_err());
    assert_eq!(char::try_from(0xE000_u32), Ok('\u{E000}'));
    assert_eq!(char::try_from(0x10FFFF_u32), Ok('\u{10FFFF}'));
    assert!(char::try_from(0x110000_u32).is_err());
    assert!(char::try_from(0xFFFF_FFFF_u32).is_err());
}

#[test]
const fn test_convert_const() {
    assert!(u32::from('a') == 0x61);
    assert!(u64::from('b') == 0x62);
    assert!(u128::from('c') == 0x63);
    assert!(char::from(b'\0') == '\0');
    assert!(char::from(b'a') == 'a');
    assert!(char::from(b'\xFF') == '\u{FF}');
}

#[test]
fn test_from_str() {
    assert_eq!(char::from_str("a").unwrap(), 'a');
    assert_eq!(char::from_str("\0").unwrap(), '\0');
    assert_eq!(char::from_str("\u{D7FF}").unwrap(), '\u{d7FF}');
    assert!(char::from_str("").is_err());
    assert!(char::from_str("abc").is_err());
}

#[test]
fn test_is_lowercase() {
    assert!('a'.is_lowercase());
    assert!('ö'.is_lowercase());
    assert!('ß'.is_lowercase());
    assert!(!'Ü'.is_lowercase());
    assert!(!'P'.is_lowercase());
}

#[test]
fn test_is_uppercase() {
    assert!(!'h'.is_uppercase());
    assert!(!'ä'.is_uppercase());
    assert!(!'ß'.is_uppercase());
    assert!('Ö'.is_uppercase());
    assert!('T'.is_uppercase());
}

#[test]
fn test_is_whitespace() {
    assert!(' '.is_whitespace());
    assert!('\u{2007}'.is_whitespace());
    assert!('\t'.is_whitespace());
    assert!('\n'.is_whitespace());
    assert!(!'a'.is_whitespace());
    assert!(!'_'.is_whitespace());
    assert!(!'\u{0}'.is_whitespace());
}

#[test]
fn test_to_digit() {
    assert_eq!('0'.to_digit(10), Some(0));
    assert_eq!('1'.to_digit(2), Some(1));
    assert_eq!('2'.to_digit(3), Some(2));
    assert_eq!('9'.to_digit(10), Some(9));
    assert_eq!('a'.to_digit(16), Some(10));
    assert_eq!('A'.to_digit(16), Some(10));
    assert_eq!('b'.to_digit(16), Some(11));
    assert_eq!('B'.to_digit(16), Some(11));
    assert_eq!('A'.to_digit(36), Some(10));
    assert_eq!('z'.to_digit(36), Some(35));
    assert_eq!('Z'.to_digit(36), Some(35));
    assert_eq!('['.to_digit(36), None);
    assert_eq!('`'.to_digit(36), None);
    assert_eq!('{'.to_digit(36), None);
    assert_eq!('$'.to_digit(36), None);
    assert_eq!('@'.to_digit(16), None);
    assert_eq!('G'.to_digit(16), None);
    assert_eq!('g'.to_digit(16), None);
    assert_eq!(' '.to_digit(10), None);
    assert_eq!('/'.to_digit(10), None);
    assert_eq!(':'.to_digit(10), None);
    assert_eq!(':'.to_digit(11), None);
}

#[test]
fn test_to_lowercase() {
    fn lower(c: char) -> String {
        let to_lowercase = c.to_lowercase();
        assert_eq!(to_lowercase.len(), to_lowercase.count());
        let iter: String = c.to_lowercase().collect();
        let disp: String = c.to_lowercase().to_string();
        assert_eq!(iter, disp);
        let iter_rev: String = c.to_lowercase().rev().collect();
        let disp_rev: String = disp.chars().rev().collect();
        assert_eq!(iter_rev, disp_rev);
        iter
    }
    assert_eq!(lower('A'), "a");
    assert_eq!(lower('Ö'), "ö");
    assert_eq!(lower('ß'), "ß");
    assert_eq!(lower('Ü'), "ü");
    assert_eq!(lower('💩'), "💩");
    assert_eq!(lower('Σ'), "σ");
    assert_eq!(lower('Τ'), "τ");
    assert_eq!(lower('Ι'), "ι");
    assert_eq!(lower('Γ'), "γ");
    assert_eq!(lower('Μ'), "μ");
    assert_eq!(lower('Α'), "α");
    assert_eq!(lower('Σ'), "σ");
    assert_eq!(lower('ǅ'), "ǆ");
    assert_eq!(lower('ﬁ'), "ﬁ");
    assert_eq!(lower('İ'), "i\u{307}");
}

#[test]
fn test_to_uppercase() {
    fn upper(c: char) -> String {
        let to_uppercase = c.to_uppercase();
        assert_eq!(to_uppercase.len(), to_uppercase.count());
        let iter: String = c.to_uppercase().collect();
        let disp: String = c.to_uppercase().to_string();
        assert_eq!(iter, disp);
        let iter_rev: String = c.to_uppercase().rev().collect();
        let disp_rev: String = disp.chars().rev().collect();
        assert_eq!(iter_rev, disp_rev);
        iter
    }
    assert_eq!(upper('a'), "A");
    assert_eq!(upper('ö'), "Ö");
    assert_eq!(upper('ß'), "SS"); // not ẞ: Latin capital letter sharp s
    assert_eq!(upper('ü'), "Ü");
    assert_eq!(upper('💩'), "💩");

    assert_eq!(upper('σ'), "Σ");
    assert_eq!(upper('τ'), "Τ");
    assert_eq!(upper('ι'), "Ι");
    assert_eq!(upper('γ'), "Γ");
    assert_eq!(upper('μ'), "Μ");
    assert_eq!(upper('α'), "Α");
    assert_eq!(upper('ς'), "Σ");
    assert_eq!(upper('ǅ'), "Ǆ");
    assert_eq!(upper('ﬁ'), "FI");
    assert_eq!(upper('ᾀ'), "ἈΙ");
}

#[test]
fn test_is_control() {
    assert!('\u{0}'.is_control());
    assert!('\u{3}'.is_control());
    assert!('\u{6}'.is_control());
    assert!('\u{9}'.is_control());
    assert!('\u{7f}'.is_control());
    assert!('\u{92}'.is_control());
    assert!(!'\u{20}'.is_control());
    assert!(!'\u{55}'.is_control());
    assert!(!'\u{68}'.is_control());
}

#[test]
fn test_is_numeric() {
    assert!('2'.is_numeric());
    assert!('7'.is_numeric());
    assert!('¾'.is_numeric());
    assert!(!'c'.is_numeric());
    assert!(!'i'.is_numeric());
    assert!(!'z'.is_numeric());
    assert!(!'Q'.is_numeric());
}

#[test]
fn test_escape_debug() {
    fn string(c: char) -> String {
        let iter: String = c.escape_debug().collect();
        let disp: String = c.escape_debug().to_string();
        assert_eq!(iter, disp);
        iter
    }
    assert_eq!(string('\n'), "\\n");
    assert_eq!(string('\r'), "\\r");
    assert_eq!(string('\''), "\\'");
    assert_eq!(string('"'), "\\\"");
    assert_eq!(string(' '), " ");
    assert_eq!(string('a'), "a");
    assert_eq!(string('~'), "~");
    assert_eq!(string('é'), "é");
    assert_eq!(string('文'), "文");
    assert_eq!(string('\x00'), "\\0");
    assert_eq!(string('\x1f'), "\\u{1f}");
    assert_eq!(string('\x7f'), "\\u{7f}");
    assert_eq!(string('\u{80}'), "\\u{80}");
    assert_eq!(string('\u{ff}'), "\u{ff}");
    assert_eq!(string('\u{11b}'), "\u{11b}");
    assert_eq!(string('\u{1d4b6}'), "\u{1d4b6}");
    assert_eq!(string('\u{301}'), "\\u{301}"); // combining character
    assert_eq!(string('\u{200b}'), "\\u{200b}"); // zero width space
    assert_eq!(string('\u{e000}'), "\\u{e000}"); // private use 1
    assert_eq!(string('\u{100000}'), "\\u{100000}"); // private use 2
}

#[test]
fn test_escape_default() {
    fn string(c: char) -> String {
        let iter: String = c.escape_default().collect();
        let disp: String = c.escape_default().to_string();
        assert_eq!(iter, disp);
        iter
    }
    assert_eq!(string('\n'), "\\n");
    assert_eq!(string('\r'), "\\r");
    assert_eq!(string('\t'), "\\t");
    assert_eq!(string('\''), "\\'");
    assert_eq!(string('"'), "\\\"");
    assert_eq!(string(' '), " ");
    assert_eq!(string('a'), "a");
    assert_eq!(string('~'), "~");
    assert_eq!(string('é'), "\\u{e9}");
    assert_eq!(string('\x00'), "\\u{0}");
    assert_eq!(string('\x1f'), "\\u{1f}");
    assert_eq!(string('\x7f'), "\\u{7f}");
    assert_eq!(string('\u{80}'), "\\u{80}");
    assert_eq!(string('\u{ff}'), "\\u{ff}");
    assert_eq!(string('\u{11b}'), "\\u{11b}");
    assert_eq!(string('\u{1d4b6}'), "\\u{1d4b6}");
    assert_eq!(string('\u{200b}'), "\\u{200b}"); // zero width space
    assert_eq!(string('\u{e000}'), "\\u{e000}"); // private use 1
    assert_eq!(string('\u{100000}'), "\\u{100000}"); // private use 2
}

#[test]
fn test_escape_unicode() {
    fn string(c: char) -> String {
        let iter: String = c.escape_unicode().collect();
        let disp: String = c.escape_unicode().to_string();
        assert_eq!(iter, disp);
        iter
    }

    assert_eq!(string('\x00'), "\\u{0}");
    assert_eq!(string('\n'), "\\u{a}");
    assert_eq!(string(' '), "\\u{20}");
    assert_eq!(string('a'), "\\u{61}");
    assert_eq!(string('\u{11b}'), "\\u{11b}");
    assert_eq!(string('\u{1d4b6}'), "\\u{1d4b6}");
}

#[test]
fn test_encode_utf8() {
    fn check(input: char, expect: &[u8]) {
        let mut buf = [0; MAX_LEN_UTF8];
        let ptr = buf.as_ptr();
        let s = input.encode_utf8(&mut buf);
        assert_eq!(s.as_ptr() as usize, ptr as usize);
        assert!(str::from_utf8(s.as_bytes()).is_ok());
        assert_eq!(s.as_bytes(), expect);
    }

    check('x', &[0x78]);
    check('\u{e9}', &[0xc3, 0xa9]);
    check('\u{a66e}', &[0xea, 0x99, 0xae]);
    check('\u{1f4a9}', &[0xf0, 0x9f, 0x92, 0xa9]);
}

#[test]
fn test_encode_utf16() {
    fn check(input: char, expect: &[u16]) {
        let mut buf = [0; 2];
        let ptr = buf.as_mut_ptr();
        let b = input.encode_utf16(&mut buf);
        assert_eq!(b.as_mut_ptr() as usize, ptr as usize);
        assert_eq!(b, expect);
    }

    check('x', &[0x0078]);
    check('\u{e9}', &[0x00e9]);
    check('\u{a66e}', &[0xa66e]);
    check('\u{1f4a9}', &[0xd83d, 0xdca9]);
}

#[test]
fn test_len_utf16() {
    assert!('x'.len_utf16() == 1);
    assert!('\u{e9}'.len_utf16() == 1);
    assert!('\u{a66e}'.len_utf16() == 1);
    assert!('\u{1f4a9}'.len_utf16() == 2);
}

#[test]
fn test_decode_utf16() {
    fn check(s: &[u16], expected: &[Result<char, u16>]) {
        let v = char::decode_utf16(s.iter().cloned())
            .map(|r| r.map_err(|e| e.unpaired_surrogate()))
            .collect::<Vec<_>>();
        assert_eq!(v, expected);
    }
    check(&[0xD800, 0x41, 0x42], &[Err(0xD800), Ok('A'), Ok('B')]);
    check(&[0xD800, 0], &[Err(0xD800), Ok('\0')]);
    check(&[0xD800], &[Err(0xD800)]);
    check(&[0xD840, 0xDC00], &[Ok('\u{20000}')]);
    check(&[0xD840, 0xD840, 0xDC00], &[Err(0xD840), Ok('\u{20000}')]);
    check(&[0xDC00, 0xD840], &[Err(0xDC00), Err(0xD840)]);
}

#[test]
fn test_decode_utf16_size_hint() {
    fn check(s: &[u16]) {
        let mut iter = char::decode_utf16(s.iter().cloned());

        loop {
            let count = iter.clone().count();
            let (lower, upper) = iter.size_hint();

            assert!(
                lower <= count && count <= upper.unwrap(),
                "lower = {lower}, count = {count}, upper = {upper:?}"
            );

            if let None = iter.next() {
                break;
            }
        }
    }

    check(&[0xD800, 0xD800, 0xDC00]);
    check(&[0xD800, 0xD800, 0x0]);
    check(&[0xD800, 0x41, 0x42]);
    check(&[0xD800, 0]);
    check(&[0xD834, 0x006d]);
}

#[test]
fn ed_iterator_specializations() {
    // Check counting
    assert_eq!('\n'.escape_default().count(), 2);
    assert_eq!('c'.escape_default().count(), 1);
    assert_eq!(' '.escape_default().count(), 1);
    assert_eq!('\\'.escape_default().count(), 2);
    assert_eq!('\''.escape_default().count(), 2);

    // Check nth

    // Check that OoB is handled correctly
    assert_eq!('\n'.escape_default().nth(2), None);
    assert_eq!('c'.escape_default().nth(1), None);
    assert_eq!(' '.escape_default().nth(1), None);
    assert_eq!('\\'.escape_default().nth(2), None);
    assert_eq!('\''.escape_default().nth(2), None);

    // Check the first char
    assert_eq!('\n'.escape_default().nth(0), Some('\\'));
    assert_eq!('c'.escape_default().nth(0), Some('c'));
    assert_eq!(' '.escape_default().nth(0), Some(' '));
    assert_eq!('\\'.escape_default().nth(0), Some('\\'));
    assert_eq!('\''.escape_default().nth(0), Some('\\'));

    // Check the second char
    assert_eq!('\n'.escape_default().nth(1), Some('n'));
    assert_eq!('\\'.escape_default().nth(1), Some('\\'));
    assert_eq!('\''.escape_default().nth(1), Some('\''));

    // Check the last char
    assert_eq!('\n'.escape_default().last(), Some('n'));
    assert_eq!('c'.escape_default().last(), Some('c'));
    assert_eq!(' '.escape_default().last(), Some(' '));
    assert_eq!('\\'.escape_default().last(), Some('\\'));
    assert_eq!('\''.escape_default().last(), Some('\''));
}

#[test]
fn eu_iterator_specializations() {
    fn check(c: char) {
        let len = c.escape_unicode().count();

        // Check OoB
        assert_eq!(c.escape_unicode().nth(len), None);

        // For all possible in-bound offsets
        let mut iter = c.escape_unicode();
        for offset in 0..len {
            // Check last
            assert_eq!(iter.clone().last(), Some('}'));

            // Check len
            assert_eq!(iter.len(), len - offset);

            // Check size_hint (= len in ExactSizeIterator)
            assert_eq!(iter.size_hint(), (iter.len(), Some(iter.len())));

            // Check counting
            assert_eq!(iter.clone().count(), len - offset);

            // Check nth
            assert_eq!(c.escape_unicode().nth(offset), iter.next());
        }

        // Check post-last
        assert_eq!(iter.clone().last(), None);
        assert_eq!(iter.clone().count(), 0);
    }

    check('\u{0}');
    check('\u{1}');
    check('\u{12}');
    check('\u{123}');
    check('\u{1234}');
    check('\u{12340}');
    check('\u{10FFFF}');
}

#[test]
#[should_panic]
fn test_from_digit_radix_too_high() {
    let _ = char::from_digit(0, 37);
}

#[test]
fn test_from_digit_invalid_radix() {
    assert!(char::from_digit(10, 9).is_none());
}

#[test]
#[should_panic]
fn test_to_digit_radix_too_low() {
    let _ = 'a'.to_digit(1);
}

#[test]
#[should_panic]
fn test_to_digit_radix_too_high() {
    let _ = 'a'.to_digit(37);
}

#[test]
fn test_as_ascii_invalid() {
    assert!('❤'.as_ascii().is_none());
}

#[test]
#[should_panic]
fn test_encode_utf8_raw_buffer_too_small() {
    let mut buf = [0u8; 1];
    let _ = char::encode_utf8_raw('ß'.into(), &mut buf);
}

#[test]
#[should_panic]
fn test_encode_utf16_raw_buffer_too_small() {
    let mut buf = [0u16; 1];
    let _ = char::encode_utf16_raw('𐐷'.into(), &mut buf);
}
