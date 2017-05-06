// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{char,str};
use std::convert::TryFrom;

#[test]
fn test_convert() {
    assert_eq!(u32::from('a'), 0x61);
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
    assert_eq!('z'.to_digit(36), Some(35));
    assert_eq!('Z'.to_digit(36), Some(35));
    assert_eq!(' '.to_digit(10), None);
    assert_eq!('$'.to_digit(36), None);
}

#[test]
fn test_to_lowercase() {
    fn lower(c: char) -> String {
        let iter: String = c.to_lowercase().collect();
        let disp: String = c.to_lowercase().to_string();
        assert_eq!(iter, disp);
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
        let iter: String = c.to_uppercase().collect();
        let disp: String = c.to_uppercase().to_string();
        assert_eq!(iter, disp);
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
fn test_is_digit() {
   assert!('2'.is_numeric());
   assert!('7'.is_numeric());
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
    assert_eq!(string('\x00'), "\\u{0}");
    assert_eq!(string('\x1f'), "\\u{1f}");
    assert_eq!(string('\x7f'), "\\u{7f}");
    assert_eq!(string('\u{80}'), "\\u{80}");
    assert_eq!(string('\u{ff}'), "\u{ff}");
    assert_eq!(string('\u{11b}'), "\u{11b}");
    assert_eq!(string('\u{1d4b6}'), "\u{1d4b6}");
    assert_eq!(string('\u{200b}'),"\\u{200b}");      // zero width space
    assert_eq!(string('\u{e000}'), "\\u{e000}");     // private use 1
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
        let mut buf = [0; 4];
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
fn test_decode_utf8() {
    macro_rules! assert_decode_utf8 {
        ($input_bytes: expr, $expected_str: expr) => {
            let input_bytes: &[u8] = &$input_bytes;
            let s = char::decode_utf8(input_bytes.iter().cloned())
                .map(|r_b| r_b.unwrap_or('\u{FFFD}'))
                .collect::<String>();
            assert_eq!(s, $expected_str,
                       "input bytes: {:?}, expected str: {:?}, result: {:?}",
                       input_bytes, $expected_str, s);
            assert_eq!(String::from_utf8_lossy(&$input_bytes), $expected_str);
        }
    }

    assert_decode_utf8!([], "");
    assert_decode_utf8!([0x41], "A");
    assert_decode_utf8!([0xC1, 0x81], "��");
    assert_decode_utf8!([0xE2, 0x99, 0xA5], "♥");
    assert_decode_utf8!([0xE2, 0x99, 0xA5, 0x41], "♥A");
    assert_decode_utf8!([0xE2, 0x99], "�");
    assert_decode_utf8!([0xE2, 0x99, 0x41], "�A");
    assert_decode_utf8!([0xC0], "�");
    assert_decode_utf8!([0xC0, 0x41], "�A");
    assert_decode_utf8!([0x80], "�");
    assert_decode_utf8!([0x80, 0x41], "�A");
    assert_decode_utf8!([0xFE], "�");
    assert_decode_utf8!([0xFE, 0x41], "�A");
    assert_decode_utf8!([0xFF], "�");
    assert_decode_utf8!([0xFF, 0x41], "�A");
    assert_decode_utf8!([0xC0, 0x80], "��");

    // Surrogates
    assert_decode_utf8!([0xED, 0x9F, 0xBF], "\u{D7FF}");
    assert_decode_utf8!([0xED, 0xA0, 0x80], "���");
    assert_decode_utf8!([0xED, 0xBF, 0x80], "���");
    assert_decode_utf8!([0xEE, 0x80, 0x80], "\u{E000}");

    // char::MAX
    assert_decode_utf8!([0xF4, 0x8F, 0xBF, 0xBF], "\u{10FFFF}");
    assert_decode_utf8!([0xF4, 0x8F, 0xBF, 0x41], "�A");
    assert_decode_utf8!([0xF4, 0x90, 0x80, 0x80], "����");

    // 5 and 6 bytes sequence
    // Part of the original design of UTF-8,
    // but invalid now that UTF-8 is artificially restricted to match the range of UTF-16.
    assert_decode_utf8!([0xF8, 0x80, 0x80, 0x80, 0x80], "�����");
    assert_decode_utf8!([0xFC, 0x80, 0x80, 0x80, 0x80, 0x80], "������");
}
