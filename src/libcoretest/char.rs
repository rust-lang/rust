// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::char::{escape_unicode, escape_default};

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
    assert!('\u2007'.is_whitespace());
    assert!('\t'.is_whitespace());
    assert!('\n'.is_whitespace());
    assert!(!'a'.is_whitespace());
    assert!(!'_'.is_whitespace());
    assert!(!'\u0000'.is_whitespace());
}

#[test]
fn test_to_digit() {
    assert_eq!('0'.to_digit(10u), Some(0u));
    assert_eq!('1'.to_digit(2u), Some(1u));
    assert_eq!('2'.to_digit(3u), Some(2u));
    assert_eq!('9'.to_digit(10u), Some(9u));
    assert_eq!('a'.to_digit(16u), Some(10u));
    assert_eq!('A'.to_digit(16u), Some(10u));
    assert_eq!('b'.to_digit(16u), Some(11u));
    assert_eq!('B'.to_digit(16u), Some(11u));
    assert_eq!('z'.to_digit(36u), Some(35u));
    assert_eq!('Z'.to_digit(36u), Some(35u));
    assert_eq!(' '.to_digit(10u), None);
    assert_eq!('$'.to_digit(36u), None);
}

#[test]
fn test_to_lowercase() {
    assert_eq!('A'.to_lowercase(), 'a');
    assert_eq!('Ö'.to_lowercase(), 'ö');
    assert_eq!('ß'.to_lowercase(), 'ß');
    assert_eq!('Ü'.to_lowercase(), 'ü');
    assert_eq!('💩'.to_lowercase(), '💩');
    assert_eq!('Σ'.to_lowercase(), 'σ');
    assert_eq!('Τ'.to_lowercase(), 'τ');
    assert_eq!('Ι'.to_lowercase(), 'ι');
    assert_eq!('Γ'.to_lowercase(), 'γ');
    assert_eq!('Μ'.to_lowercase(), 'μ');
    assert_eq!('Α'.to_lowercase(), 'α');
    assert_eq!('Σ'.to_lowercase(), 'σ');
}

#[test]
fn test_to_uppercase() {
    assert_eq!('a'.to_uppercase(), 'A');
    assert_eq!('ö'.to_uppercase(), 'Ö');
    assert_eq!('ß'.to_uppercase(), 'ß'); // not ẞ: Latin capital letter sharp s
    assert_eq!('ü'.to_uppercase(), 'Ü');
    assert_eq!('💩'.to_uppercase(), '💩');

    assert_eq!('σ'.to_uppercase(), 'Σ');
    assert_eq!('τ'.to_uppercase(), 'Τ');
    assert_eq!('ι'.to_uppercase(), 'Ι');
    assert_eq!('γ'.to_uppercase(), 'Γ');
    assert_eq!('μ'.to_uppercase(), 'Μ');
    assert_eq!('α'.to_uppercase(), 'Α');
    assert_eq!('ς'.to_uppercase(), 'Σ');
}

#[test]
fn test_is_control() {
    assert!('\u0000'.is_control());
    assert!('\u0003'.is_control());
    assert!('\u0006'.is_control());
    assert!('\u0009'.is_control());
    assert!('\u007f'.is_control());
    assert!('\u0092'.is_control());
    assert!(!'\u0020'.is_control());
    assert!(!'\u0055'.is_control());
    assert!(!'\u0068'.is_control());
}

#[test]
fn test_is_digit() {
   assert!('2'.is_digit());
   assert!('7'.is_digit());
   assert!(!'c'.is_digit());
   assert!(!'i'.is_digit());
   assert!(!'z'.is_digit());
   assert!(!'Q'.is_digit());
}

#[test]
fn test_escape_default() {
    fn string(c: char) -> String {
        let mut result = String::new();
        escape_default(c, |c| { result.push_char(c); });
        return result;
    }
    let s = string('\n');
    assert_eq!(s.as_slice(), "\\n");
    let s = string('\r');
    assert_eq!(s.as_slice(), "\\r");
    let s = string('\'');
    assert_eq!(s.as_slice(), "\\'");
    let s = string('"');
    assert_eq!(s.as_slice(), "\\\"");
    let s = string(' ');
    assert_eq!(s.as_slice(), " ");
    let s = string('a');
    assert_eq!(s.as_slice(), "a");
    let s = string('~');
    assert_eq!(s.as_slice(), "~");
    let s = string('\x00');
    assert_eq!(s.as_slice(), "\\x00");
    let s = string('\x1f');
    assert_eq!(s.as_slice(), "\\x1f");
    let s = string('\x7f');
    assert_eq!(s.as_slice(), "\\x7f");
    let s = string('\xff');
    assert_eq!(s.as_slice(), "\\xff");
    let s = string('\u011b');
    assert_eq!(s.as_slice(), "\\u011b");
    let s = string('\U0001d4b6');
    assert_eq!(s.as_slice(), "\\U0001d4b6");
}

#[test]
fn test_escape_unicode() {
    fn string(c: char) -> String {
        let mut result = String::new();
        escape_unicode(c, |c| { result.push_char(c); });
        return result;
    }
    let s = string('\x00');
    assert_eq!(s.as_slice(), "\\x00");
    let s = string('\n');
    assert_eq!(s.as_slice(), "\\x0a");
    let s = string(' ');
    assert_eq!(s.as_slice(), "\\x20");
    let s = string('a');
    assert_eq!(s.as_slice(), "\\x61");
    let s = string('\u011b');
    assert_eq!(s.as_slice(), "\\u011b");
    let s = string('\U0001d4b6');
    assert_eq!(s.as_slice(), "\\U0001d4b6");
}

#[test]
fn test_encode_utf8() {
    fn check(input: char, expect: &[u8]) {
        let mut buf = [0u8, ..4];
        let n = input.encode_utf8(buf /* as mut slice! */);
        assert_eq!(buf.slice_to(n), expect);
    }

    check('x', [0x78]);
    check('\u00e9', [0xc3, 0xa9]);
    check('\ua66e', [0xea, 0x99, 0xae]);
    check('\U0001f4a9', [0xf0, 0x9f, 0x92, 0xa9]);
}

#[test]
fn test_encode_utf16() {
    fn check(input: char, expect: &[u16]) {
        let mut buf = [0u16, ..2];
        let n = input.encode_utf16(buf /* as mut slice! */);
        assert_eq!(buf.slice_to(n), expect);
    }

    check('x', [0x0078]);
    check('\u00e9', [0x00e9]);
    check('\ua66e', [0xa66e]);
    check('\U0001f4a9', [0xd83d, 0xdca9]);
}
