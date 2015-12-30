// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    assert!(!'\e'.is_whitespace());
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
    fn lower(c: char) -> Vec<char> {
        c.to_lowercase().collect()
    }
    assert_eq!(lower('A'), ['a']);
    assert_eq!(lower('Ö'), ['ö']);
    assert_eq!(lower('ß'), ['ß']);
    assert_eq!(lower('Ü'), ['ü']);
    assert_eq!(lower('💩'), ['💩']);
    assert_eq!(lower('Σ'), ['σ']);
    assert_eq!(lower('Τ'), ['τ']);
    assert_eq!(lower('Ι'), ['ι']);
    assert_eq!(lower('Γ'), ['γ']);
    assert_eq!(lower('Μ'), ['μ']);
    assert_eq!(lower('Α'), ['α']);
    assert_eq!(lower('Σ'), ['σ']);
    assert_eq!(lower('ǅ'), ['ǆ']);
    assert_eq!(lower('ﬁ'), ['ﬁ']);
    assert_eq!(lower('İ'), ['i', '\u{307}']);
}

#[test]
fn test_to_uppercase() {
    fn upper(c: char) -> Vec<char> {
        c.to_uppercase().collect()
    }
    assert_eq!(upper('a'), ['A']);
    assert_eq!(upper('ö'), ['Ö']);
    assert_eq!(upper('ß'), ['S', 'S']); // not ẞ: Latin capital letter sharp s
    assert_eq!(upper('ü'), ['Ü']);
    assert_eq!(upper('💩'), ['💩']);

    assert_eq!(upper('σ'), ['Σ']);
    assert_eq!(upper('τ'), ['Τ']);
    assert_eq!(upper('ι'), ['Ι']);
    assert_eq!(upper('γ'), ['Γ']);
    assert_eq!(upper('μ'), ['Μ']);
    assert_eq!(upper('α'), ['Α']);
    assert_eq!(upper('ς'), ['Σ']);
    assert_eq!(upper('ǅ'), ['Ǆ']);
    assert_eq!(upper('ﬁ'), ['F', 'I']);
    assert_eq!(upper('ᾀ'), ['Ἀ', 'Ι']);
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
fn test_escape_default() {
    fn string(c: char) -> String {
        c.escape_default().collect()
    }
    let s = string('\n');
    assert_eq!(s, "\\n");
    let s = string('\r');
    assert_eq!(s, "\\r");
    let s = string('\'');
    assert_eq!(s, "\\'");
    let s = string('"');
    assert_eq!(s, "\\\"");
    let s = string(' ');
    assert_eq!(s, " ");
    let s = string('a');
    assert_eq!(s, "a");
    let s = string('~');
    assert_eq!(s, "~");
    let s = string('\x00');
    assert_eq!(s, "\\u{0}");
    let s = string('\x1f');
    assert_eq!(s, "\\u{1f}");
    let s = string('\x7f');
    assert_eq!(s, "\\u{7f}");
    let s = string('\u{ff}');
    assert_eq!(s, "\\u{ff}");
    let s = string('\u{11b}');
    assert_eq!(s, "\\u{11b}");
    let s = string('\u{1d4b6}');
    assert_eq!(s, "\\u{1d4b6}");
}

#[test]
fn test_escape_unicode() {
    fn string(c: char) -> String { c.escape_unicode().collect() }

    let s = string('\x00');
    assert_eq!(s, "\\u{0}");
    let s = string('\n');
    assert_eq!(s, "\\u{a}");
    let s = string(' ');
    assert_eq!(s, "\\u{20}");
    let s = string('a');
    assert_eq!(s, "\\u{61}");
    let s = string('\u{11b}');
    assert_eq!(s, "\\u{11b}");
    let s = string('\u{1d4b6}');
    assert_eq!(s, "\\u{1d4b6}");
}

#[test]
fn test_encode_utf8() {
    fn check(input: char, expect: &[u8]) {
        let mut buf = [0; 4];
        let n = input.encode_utf8(&mut buf).unwrap_or(0);
        assert_eq!(&buf[..n], expect);
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
        let n = input.encode_utf16(&mut buf).unwrap_or(0);
        assert_eq!(&buf[..n], expect);
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
        assert_eq!(::std::char::decode_utf16(s.iter().cloned()).collect::<Vec<_>>(), expected);
    }
    check(&[0xD800, 0x41, 0x42], &[Err(0xD800), Ok('A'), Ok('B')]);
    check(&[0xD800, 0], &[Err(0xD800), Ok('\0')]);
}
