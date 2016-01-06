// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering::{Equal, Greater, Less};
use std::str::from_utf8;

#[test]
fn test_le() {
    assert!("" <= "");
    assert!("" <= "foo");
    assert!("foo" <= "foo");
    assert!("foo" != "bar");
}

#[test]
fn test_find() {
    assert_eq!("hello".find('l'), Some(2));
    assert_eq!("hello".find(|c:char| c == 'o'), Some(4));
    assert!("hello".find('x').is_none());
    assert!("hello".find(|c:char| c == 'x').is_none());
    assert_eq!("ประเทศไทย中华Việt Nam".find('华'), Some(30));
    assert_eq!("ประเทศไทย中华Việt Nam".find(|c: char| c == '华'), Some(30));
}

#[test]
fn test_rfind() {
    assert_eq!("hello".rfind('l'), Some(3));
    assert_eq!("hello".rfind(|c:char| c == 'o'), Some(4));
    assert!("hello".rfind('x').is_none());
    assert!("hello".rfind(|c:char| c == 'x').is_none());
    assert_eq!("ประเทศไทย中华Việt Nam".rfind('华'), Some(30));
    assert_eq!("ประเทศไทย中华Việt Nam".rfind(|c: char| c == '华'), Some(30));
}

#[test]
fn test_collect() {
    let empty = "";
    let s: String = empty.chars().collect();
    assert_eq!(empty, s);
    let data = "ประเทศไทย中";
    let s: String = data.chars().collect();
    assert_eq!(data, s);
}

#[test]
fn test_into_bytes() {
    let data = String::from("asdf");
    let buf = data.into_bytes();
    assert_eq!(buf, b"asdf");
}

#[test]
fn test_find_str() {
    // byte positions
    assert_eq!("".find(""), Some(0));
    assert!("banana".find("apple pie").is_none());

    let data = "abcabc";
    assert_eq!(data[0..6].find("ab"), Some(0));
    assert_eq!(data[2..6].find("ab"), Some(3 - 2));
    assert!(data[2..4].find("ab").is_none());

    let string = "ประเทศไทย中华Việt Nam";
    let mut data = String::from(string);
    data.push_str(string);
    assert!(data.find("ไท华").is_none());
    assert_eq!(data[0..43].find(""), Some(0));
    assert_eq!(data[6..43].find(""), Some(6 - 6));

    assert_eq!(data[0..43].find("ประ"), Some( 0));
    assert_eq!(data[0..43].find("ทศไ"), Some(12));
    assert_eq!(data[0..43].find("ย中"), Some(24));
    assert_eq!(data[0..43].find("iệt"), Some(34));
    assert_eq!(data[0..43].find("Nam"), Some(40));

    assert_eq!(data[43..86].find("ประ"), Some(43 - 43));
    assert_eq!(data[43..86].find("ทศไ"), Some(55 - 43));
    assert_eq!(data[43..86].find("ย中"), Some(67 - 43));
    assert_eq!(data[43..86].find("iệt"), Some(77 - 43));
    assert_eq!(data[43..86].find("Nam"), Some(83 - 43));

    // find every substring -- assert that it finds it, or an earlier occurrence.
    let string = "Việt Namacbaabcaabaaba";
    for (i, ci) in string.char_indices() {
        let ip = i + ci.len_utf8();
        for j in string[ip..].char_indices()
                             .map(|(i, _)| i)
                             .chain(Some(string.len() - ip))
        {
            let pat = &string[i..ip + j];
            assert!(match string.find(pat) {
                None => false,
                Some(x) => x <= i,
            });
            assert!(match string.rfind(pat) {
                None => false,
                Some(x) => x >= i,
            });
        }
    }
}

fn s(x: &str) -> String { x.to_string() }

macro_rules! test_concat {
    ($expected: expr, $string: expr) => {
        {
            let s: String = $string.concat();
            assert_eq!($expected, s);
        }
    }
}

#[test]
fn test_concat_for_different_types() {
    test_concat!("ab", vec![s("a"), s("b")]);
    test_concat!("ab", vec!["a", "b"]);
    test_concat!("ab", vec!["a", "b"]);
    test_concat!("ab", vec![s("a"), s("b")]);
}

#[test]
fn test_concat_for_different_lengths() {
    let empty: &[&str] = &[];
    test_concat!("", empty);
    test_concat!("a", ["a"]);
    test_concat!("ab", ["a", "b"]);
    test_concat!("abc", ["", "a", "bc"]);
}

macro_rules! test_join {
    ($expected: expr, $string: expr, $delim: expr) => {
        {
            let s = $string.join($delim);
            assert_eq!($expected, s);
        }
    }
}

#[test]
fn test_join_for_different_types() {
    test_join!("a-b", ["a", "b"], "-");
    let hyphen = "-".to_string();
    test_join!("a-b", [s("a"), s("b")], &*hyphen);
    test_join!("a-b", vec!["a", "b"], &*hyphen);
    test_join!("a-b", &*vec!["a", "b"], "-");
    test_join!("a-b", vec![s("a"), s("b")], "-");
}

#[test]
fn test_join_for_different_lengths() {
    let empty: &[&str] = &[];
    test_join!("", empty, "-");
    test_join!("a", ["a"], "-");
    test_join!("a-b", ["a", "b"], "-");
    test_join!("-a-bc", ["", "a", "bc"], "-");
}

#[test]
fn test_unsafe_slice() {
    assert_eq!("ab", unsafe {"abc".slice_unchecked(0, 2)});
    assert_eq!("bc", unsafe {"abc".slice_unchecked(1, 3)});
    assert_eq!("", unsafe {"abc".slice_unchecked(1, 1)});
    fn a_million_letter_a() -> String {
        let mut i = 0;
        let mut rs = String::new();
        while i < 100000 {
            rs.push_str("aaaaaaaaaa");
            i += 1;
        }
        rs
    }
    fn half_a_million_letter_a() -> String {
        let mut i = 0;
        let mut rs = String::new();
        while i < 100000 {
            rs.push_str("aaaaa");
            i += 1;
        }
        rs
    }
    let letters = a_million_letter_a();
    assert_eq!(half_a_million_letter_a(),
        unsafe { letters.slice_unchecked(0, 500000)});
}

#[test]
fn test_starts_with() {
    assert!(("".starts_with("")));
    assert!(("abc".starts_with("")));
    assert!(("abc".starts_with("a")));
    assert!((!"a".starts_with("abc")));
    assert!((!"".starts_with("abc")));
    assert!((!"ödd".starts_with("-")));
    assert!(("ödd".starts_with("öd")));
}

#[test]
fn test_ends_with() {
    assert!(("".ends_with("")));
    assert!(("abc".ends_with("")));
    assert!(("abc".ends_with("c")));
    assert!((!"a".ends_with("abc")));
    assert!((!"".ends_with("abc")));
    assert!((!"ddö".ends_with("-")));
    assert!(("ddö".ends_with("dö")));
}

#[test]
fn test_is_empty() {
    assert!("".is_empty());
    assert!(!"a".is_empty());
}

#[test]
fn test_replace() {
    let a = "a";
    assert_eq!("".replace(a, "b"), "");
    assert_eq!("a".replace(a, "b"), "b");
    assert_eq!("ab".replace(a, "b"), "bb");
    let test = "test";
    assert_eq!(" test test ".replace(test, "toast"), " toast toast ");
    assert_eq!(" test test ".replace(test, ""), "   ");
}

#[test]
fn test_replace_2a() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let a = "ประเ";
    let a2 = "دولة الكويتทศไทย中华";
    assert_eq!(data.replace(a, repl), a2);
}

#[test]
fn test_replace_2b() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let b = "ะเ";
    let b2 = "ปรدولة الكويتทศไทย中华";
    assert_eq!(data.replace(b, repl), b2);
}

#[test]
fn test_replace_2c() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let c = "中华";
    let c2 = "ประเทศไทยدولة الكويت";
    assert_eq!(data.replace(c, repl), c2);
}

#[test]
fn test_replace_2d() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let d = "ไท华";
    assert_eq!(data.replace(d, repl), data);
}

#[test]
fn test_slice() {
    assert_eq!("ab", &"abc"[0..2]);
    assert_eq!("bc", &"abc"[1..3]);
    assert_eq!("", &"abc"[1..1]);
    assert_eq!("\u{65e5}", &"\u{65e5}\u{672c}"[0..3]);

    let data = "ประเทศไทย中华";
    assert_eq!("ป", &data[0..3]);
    assert_eq!("ร", &data[3..6]);
    assert_eq!("", &data[3..3]);
    assert_eq!("华", &data[30..33]);

    fn a_million_letter_x() -> String {
        let mut i = 0;
        let mut rs = String::new();
        while i < 100000 {
            rs.push_str("华华华华华华华华华华");
            i += 1;
        }
        rs
    }
    fn half_a_million_letter_x() -> String {
        let mut i = 0;
        let mut rs = String::new();
        while i < 100000 {
            rs.push_str("华华华华华");
            i += 1;
        }
        rs
    }
    let letters = a_million_letter_x();
    assert_eq!(half_a_million_letter_x(), &letters[0..3 * 500000]);
}

#[test]
fn test_slice_2() {
    let ss = "中华Việt Nam";

    assert_eq!("华", &ss[3..6]);
    assert_eq!("Việt Nam", &ss[6..16]);

    assert_eq!("ab", &"abc"[0..2]);
    assert_eq!("bc", &"abc"[1..3]);
    assert_eq!("", &"abc"[1..1]);

    assert_eq!("中", &ss[0..3]);
    assert_eq!("华V", &ss[3..7]);
    assert_eq!("", &ss[3..3]);
    /*0: 中
      3: 华
      6: V
      7: i
      8: ệ
     11: t
     12:
     13: N
     14: a
     15: m */
}

#[test]
#[should_panic]
fn test_slice_fail() {
    &"中华Việt Nam"[0..2];
}

#[test]
fn test_slice_from() {
    assert_eq!(&"abcd"[0..], "abcd");
    assert_eq!(&"abcd"[2..], "cd");
    assert_eq!(&"abcd"[4..], "");
}
#[test]
fn test_slice_to() {
    assert_eq!(&"abcd"[..0], "");
    assert_eq!(&"abcd"[..2], "ab");
    assert_eq!(&"abcd"[..4], "abcd");
}

#[test]
fn test_trim_left_matches() {
    let v: &[char] = &[];
    assert_eq!(" *** foo *** ".trim_left_matches(v), " *** foo *** ");
    let chars: &[char] = &['*', ' '];
    assert_eq!(" *** foo *** ".trim_left_matches(chars), "foo *** ");
    assert_eq!(" ***  *** ".trim_left_matches(chars), "");
    assert_eq!("foo *** ".trim_left_matches(chars), "foo *** ");

    assert_eq!("11foo1bar11".trim_left_matches('1'), "foo1bar11");
    let chars: &[char] = &['1', '2'];
    assert_eq!("12foo1bar12".trim_left_matches(chars), "foo1bar12");
    assert_eq!("123foo1bar123".trim_left_matches(|c: char| c.is_numeric()), "foo1bar123");
}

#[test]
fn test_trim_right_matches() {
    let v: &[char] = &[];
    assert_eq!(" *** foo *** ".trim_right_matches(v), " *** foo *** ");
    let chars: &[char] = &['*', ' '];
    assert_eq!(" *** foo *** ".trim_right_matches(chars), " *** foo");
    assert_eq!(" ***  *** ".trim_right_matches(chars), "");
    assert_eq!(" *** foo".trim_right_matches(chars), " *** foo");

    assert_eq!("11foo1bar11".trim_right_matches('1'), "11foo1bar");
    let chars: &[char] = &['1', '2'];
    assert_eq!("12foo1bar12".trim_right_matches(chars), "12foo1bar");
    assert_eq!("123foo1bar123".trim_right_matches(|c: char| c.is_numeric()), "123foo1bar");
}

#[test]
fn test_trim_matches() {
    let v: &[char] = &[];
    assert_eq!(" *** foo *** ".trim_matches(v), " *** foo *** ");
    let chars: &[char] = &['*', ' '];
    assert_eq!(" *** foo *** ".trim_matches(chars), "foo");
    assert_eq!(" ***  *** ".trim_matches(chars), "");
    assert_eq!("foo".trim_matches(chars), "foo");

    assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    let chars: &[char] = &['1', '2'];
    assert_eq!("12foo1bar12".trim_matches(chars), "foo1bar");
    assert_eq!("123foo1bar123".trim_matches(|c: char| c.is_numeric()), "foo1bar");
}

#[test]
fn test_trim_left() {
    assert_eq!("".trim_left(), "");
    assert_eq!("a".trim_left(), "a");
    assert_eq!("    ".trim_left(), "");
    assert_eq!("     blah".trim_left(), "blah");
    assert_eq!("   \u{3000}  wut".trim_left(), "wut");
    assert_eq!("hey ".trim_left(), "hey ");
}

#[test]
fn test_trim_right() {
    assert_eq!("".trim_right(), "");
    assert_eq!("a".trim_right(), "a");
    assert_eq!("    ".trim_right(), "");
    assert_eq!("blah     ".trim_right(), "blah");
    assert_eq!("wut   \u{3000}  ".trim_right(), "wut");
    assert_eq!(" hey".trim_right(), " hey");
}

#[test]
fn test_trim() {
    assert_eq!("".trim(), "");
    assert_eq!("a".trim(), "a");
    assert_eq!("    ".trim(), "");
    assert_eq!("    blah     ".trim(), "blah");
    assert_eq!("\nwut   \u{3000}  ".trim(), "wut");
    assert_eq!(" hey dude ".trim(), "hey dude");
}

#[test]
fn test_is_whitespace() {
    assert!("".chars().all(|c| c.is_whitespace()));
    assert!(" ".chars().all(|c| c.is_whitespace()));
    assert!("\u{2009}".chars().all(|c| c.is_whitespace())); // Thin space
    assert!("  \n\t   ".chars().all(|c| c.is_whitespace()));
    assert!(!"   _   ".chars().all(|c| c.is_whitespace()));
}

#[test]
fn test_slice_shift_char() {
    let data = "ประเทศไทย中";
    assert_eq!(data.slice_shift_char(), Some(('ป', "ระเทศไทย中")));
}

#[test]
fn test_slice_shift_char_2() {
    let empty = "";
    assert_eq!(empty.slice_shift_char(), None);
}

#[test]
fn test_is_utf8() {
    // deny overlong encodings
    assert!(from_utf8(&[0xc0, 0x80]).is_err());
    assert!(from_utf8(&[0xc0, 0xae]).is_err());
    assert!(from_utf8(&[0xe0, 0x80, 0x80]).is_err());
    assert!(from_utf8(&[0xe0, 0x80, 0xaf]).is_err());
    assert!(from_utf8(&[0xe0, 0x81, 0x81]).is_err());
    assert!(from_utf8(&[0xf0, 0x82, 0x82, 0xac]).is_err());
    assert!(from_utf8(&[0xf4, 0x90, 0x80, 0x80]).is_err());

    // deny surrogates
    assert!(from_utf8(&[0xED, 0xA0, 0x80]).is_err());
    assert!(from_utf8(&[0xED, 0xBF, 0xBF]).is_err());

    assert!(from_utf8(&[0xC2, 0x80]).is_ok());
    assert!(from_utf8(&[0xDF, 0xBF]).is_ok());
    assert!(from_utf8(&[0xE0, 0xA0, 0x80]).is_ok());
    assert!(from_utf8(&[0xED, 0x9F, 0xBF]).is_ok());
    assert!(from_utf8(&[0xEE, 0x80, 0x80]).is_ok());
    assert!(from_utf8(&[0xEF, 0xBF, 0xBF]).is_ok());
    assert!(from_utf8(&[0xF0, 0x90, 0x80, 0x80]).is_ok());
    assert!(from_utf8(&[0xF4, 0x8F, 0xBF, 0xBF]).is_ok());
}

#[test]
fn from_utf8_mostly_ascii() {
    // deny invalid bytes embedded in long stretches of ascii
    for i in 32..64 {
        let mut data = [0; 128];
        data[i] = 0xC0;
        assert!(from_utf8(&data).is_err());
        data[i] = 0xC2;
        assert!(from_utf8(&data).is_err());
    }
}

#[test]
fn test_is_utf16() {
    use rustc_unicode::str::is_utf16;

    macro_rules! pos {
        ($($e:expr),*) => { { $(assert!(is_utf16($e));)* } }
    }

    // non-surrogates
    pos!(&[0x0000],
         &[0x0001, 0x0002],
         &[0xD7FF],
         &[0xE000]);

    // surrogate pairs (randomly generated with Python 3's
    // .encode('utf-16be'))
    pos!(&[0xdb54, 0xdf16, 0xd880, 0xdee0, 0xdb6a, 0xdd45],
         &[0xd91f, 0xdeb1, 0xdb31, 0xdd84, 0xd8e2, 0xde14],
         &[0xdb9f, 0xdc26, 0xdb6f, 0xde58, 0xd850, 0xdfae]);

    // mixtures (also random)
    pos!(&[0xd921, 0xdcc2, 0x002d, 0x004d, 0xdb32, 0xdf65],
         &[0xdb45, 0xdd2d, 0x006a, 0xdacd, 0xddfe, 0x0006],
         &[0x0067, 0xd8ff, 0xddb7, 0x000f, 0xd900, 0xdc80]);

    // negative tests
    macro_rules! neg {
        ($($e:expr),*) => { { $(assert!(!is_utf16($e));)* } }
    }

    neg!(
        // surrogate + regular unit
        &[0xdb45, 0x0000],
        // surrogate + lead surrogate
        &[0xd900, 0xd900],
        // unterminated surrogate
        &[0xd8ff],
        // trail surrogate without a lead
        &[0xddb7]);

    // random byte sequences that Python 3's .decode('utf-16be')
    // failed on
    neg!(&[0x5b3d, 0x0141, 0xde9e, 0x8fdc, 0xc6e7],
         &[0xdf5a, 0x82a5, 0x62b9, 0xb447, 0x92f3],
         &[0xda4e, 0x42bc, 0x4462, 0xee98, 0xc2ca],
         &[0xbe00, 0xb04a, 0x6ecb, 0xdd89, 0xe278],
         &[0x0465, 0xab56, 0xdbb6, 0xa893, 0x665e],
         &[0x6b7f, 0x0a19, 0x40f4, 0xa657, 0xdcc5],
         &[0x9b50, 0xda5e, 0x24ec, 0x03ad, 0x6dee],
         &[0x8d17, 0xcaa7, 0xf4ae, 0xdf6e, 0xbed7],
         &[0xdaee, 0x2584, 0x7d30, 0xa626, 0x121a],
         &[0xd956, 0x4b43, 0x7570, 0xccd6, 0x4f4a],
         &[0x9dcf, 0x1b49, 0x4ba5, 0xfce9, 0xdffe],
         &[0x6572, 0xce53, 0xb05a, 0xf6af, 0xdacf],
         &[0x1b90, 0x728c, 0x9906, 0xdb68, 0xf46e],
         &[0x1606, 0xbeca, 0xbe76, 0x860f, 0xdfa5],
         &[0x8b4f, 0xde7a, 0xd220, 0x9fac, 0x2b6f],
         &[0xb8fe, 0xebbe, 0xda32, 0x1a5f, 0x8b8b],
         &[0x934b, 0x8956, 0xc434, 0x1881, 0xddf7],
         &[0x5a95, 0x13fc, 0xf116, 0xd89b, 0x93f9],
         &[0xd640, 0x71f1, 0xdd7d, 0x77eb, 0x1cd8],
         &[0x348b, 0xaef0, 0xdb2c, 0xebf1, 0x1282],
         &[0x50d7, 0xd824, 0x5010, 0xb369, 0x22ea]);
}

#[test]
fn test_as_bytes() {
    // no null
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
        184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
        109
    ];
    let b: &[u8] = &[];
    assert_eq!("".as_bytes(), b);
    assert_eq!("abc".as_bytes(), b"abc");
    assert_eq!("ศไทย中华Việt Nam".as_bytes(), v);
}

#[test]
#[should_panic]
fn test_as_bytes_fail() {
    // Don't double free. (I'm not sure if this exercises the
    // original problem code path anymore.)
    let s = String::from("");
    let _bytes = s.as_bytes();
    panic!();
}

#[test]
fn test_as_ptr() {
    let buf = "hello".as_ptr();
    unsafe {
        assert_eq!(*buf.offset(0), b'h');
        assert_eq!(*buf.offset(1), b'e');
        assert_eq!(*buf.offset(2), b'l');
        assert_eq!(*buf.offset(3), b'l');
        assert_eq!(*buf.offset(4), b'o');
    }
}

#[test]
fn vec_str_conversions() {
    let s1: String = String::from("All mimsy were the borogoves");

    let v: Vec<u8> = s1.as_bytes().to_vec();
    let s2: String = String::from(from_utf8(&v).unwrap());
    let mut i = 0;
    let n1 = s1.len();
    let n2 = v.len();
    assert_eq!(n1, n2);
    while i < n1 {
        let a: u8 = s1.as_bytes()[i];
        let b: u8 = s2.as_bytes()[i];
        debug!("{}", a);
        debug!("{}", b);
        assert_eq!(a, b);
        i += 1;
    }
}

#[test]
fn test_contains() {
    assert!("abcde".contains("bcd"));
    assert!("abcde".contains("abcd"));
    assert!("abcde".contains("bcde"));
    assert!("abcde".contains(""));
    assert!("".contains(""));
    assert!(!"abcde".contains("def"));
    assert!(!"".contains("a"));

    let data = "ประเทศไทย中华Việt Nam";
    assert!(data.contains("ประเ"));
    assert!(data.contains("ะเ"));
    assert!(data.contains("中华"));
    assert!(!data.contains("ไท华"));
}

#[test]
fn test_contains_char() {
    assert!("abc".contains('b'));
    assert!("a".contains('a'));
    assert!(!"abc".contains('d'));
    assert!(!"".contains('a'));
}

#[test]
fn test_char_at() {
    let s = "ศไทย中华Việt Nam";
    let v = vec!['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
    let mut pos = 0;
    for ch in &v {
        assert!(s.char_at(pos) == *ch);
        pos += ch.to_string().len();
    }
}

#[test]
fn test_char_at_reverse() {
    let s = "ศไทย中华Việt Nam";
    let v = vec!['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
    let mut pos = s.len();
    for ch in v.iter().rev() {
        assert!(s.char_at_reverse(pos) == *ch);
        pos -= ch.to_string().len();
    }
}

#[test]
fn test_split_at() {
    let s = "ศไทย中华Việt Nam";
    for (index, _) in s.char_indices() {
        let (a, b) = s.split_at(index);
        assert_eq!(&s[..a.len()], a);
        assert_eq!(&s[a.len()..], b);
    }
    let (a, b) = s.split_at(s.len());
    assert_eq!(a, s);
    assert_eq!(b, "");
}

#[test]
fn test_split_at_mut() {
    use std::ascii::AsciiExt;
    let mut s = "Hello World".to_string();
    {
        let (a, b) = s.split_at_mut(5);
        a.make_ascii_uppercase();
        b.make_ascii_lowercase();
    }
    assert_eq!(s, "HELLO world");
}

#[test]
#[should_panic]
fn test_split_at_boundscheck() {
    let s = "ศไทย中华Việt Nam";
    s.split_at(1);
}

#[test]
fn test_escape_unicode() {
    assert_eq!("abc".escape_unicode(), "\\u{61}\\u{62}\\u{63}");
    assert_eq!("a c".escape_unicode(), "\\u{61}\\u{20}\\u{63}");
    assert_eq!("\r\n\t".escape_unicode(), "\\u{d}\\u{a}\\u{9}");
    assert_eq!("'\"\\".escape_unicode(), "\\u{27}\\u{22}\\u{5c}");
    assert_eq!("\x00\x01\u{fe}\u{ff}".escape_unicode(), "\\u{0}\\u{1}\\u{fe}\\u{ff}");
    assert_eq!("\u{100}\u{ffff}".escape_unicode(), "\\u{100}\\u{ffff}");
    assert_eq!("\u{10000}\u{10ffff}".escape_unicode(), "\\u{10000}\\u{10ffff}");
    assert_eq!("ab\u{fb00}".escape_unicode(), "\\u{61}\\u{62}\\u{fb00}");
    assert_eq!("\u{1d4ea}\r".escape_unicode(), "\\u{1d4ea}\\u{d}");
}

#[test]
fn test_escape_default() {
    assert_eq!("abc".escape_default(), "abc");
    assert_eq!("a c".escape_default(), "a c");
    assert_eq!("\r\n\t".escape_default(), "\\r\\n\\t");
    assert_eq!("'\"\\".escape_default(), "\\'\\\"\\\\");
    assert_eq!("\u{100}\u{ffff}".escape_default(), "\\u{100}\\u{ffff}");
    assert_eq!("\u{10000}\u{10ffff}".escape_default(), "\\u{10000}\\u{10ffff}");
    assert_eq!("ab\u{fb00}".escape_default(), "ab\\u{fb00}");
    assert_eq!("\u{1d4ea}\r".escape_default(), "\\u{1d4ea}\\r");
}

#[test]
fn test_total_ord() {
    assert_eq!("1234".cmp("123"), Greater);
    assert_eq!("123".cmp("1234"), Less);
    assert_eq!("1234".cmp("1234"), Equal);
    assert_eq!("12345555".cmp("123456"), Less);
    assert_eq!("22".cmp("1234"), Greater);
}

#[test]
fn test_char_range_at() {
    let data = "b¢€𤭢𤭢€¢b";
    assert_eq!('b', data.char_range_at(0).ch);
    assert_eq!('¢', data.char_range_at(1).ch);
    assert_eq!('€', data.char_range_at(3).ch);
    assert_eq!('𤭢', data.char_range_at(6).ch);
    assert_eq!('𤭢', data.char_range_at(10).ch);
    assert_eq!('€', data.char_range_at(14).ch);
    assert_eq!('¢', data.char_range_at(17).ch);
    assert_eq!('b', data.char_range_at(19).ch);
}

#[test]
fn test_char_range_at_reverse_underflow() {
    assert_eq!("abc".char_range_at_reverse(0).next, 0);
}

#[test]
fn test_iterator() {
    let s = "ศไทย中华Việt Nam";
    let v = ['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];

    let mut pos = 0;
    let it = s.chars();

    for c in it {
        assert_eq!(c, v[pos]);
        pos += 1;
    }
    assert_eq!(pos, v.len());
}

#[test]
fn test_rev_iterator() {
    let s = "ศไทย中华Việt Nam";
    let v = ['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

    let mut pos = 0;
    let it = s.chars().rev();

    for c in it {
        assert_eq!(c, v[pos]);
        pos += 1;
    }
    assert_eq!(pos, v.len());
}

#[test]
fn test_chars_decoding() {
    let mut bytes = [0; 4];
    for c in (0..0x110000).filter_map(::std::char::from_u32) {
        let len = c.encode_utf8(&mut bytes).unwrap_or(0);
        let s = ::std::str::from_utf8(&bytes[..len]).unwrap();
        if Some(c) != s.chars().next() {
            panic!("character {:x}={} does not decode correctly", c as u32, c);
        }
    }
}

#[test]
fn test_chars_rev_decoding() {
    let mut bytes = [0; 4];
    for c in (0..0x110000).filter_map(::std::char::from_u32) {
        let len = c.encode_utf8(&mut bytes).unwrap_or(0);
        let s = ::std::str::from_utf8(&bytes[..len]).unwrap();
        if Some(c) != s.chars().rev().next() {
            panic!("character {:x}={} does not decode correctly", c as u32, c);
        }
    }
}

#[test]
fn test_iterator_clone() {
    let s = "ศไทย中华Việt Nam";
    let mut it = s.chars();
    it.next();
    assert!(it.clone().zip(it).all(|(x,y)| x == y));
}

#[test]
fn test_bytesator() {
    let s = "ศไทย中华Việt Nam";
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
        184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
        109
    ];
    let mut pos = 0;

    for b in s.bytes() {
        assert_eq!(b, v[pos]);
        pos += 1;
    }
}

#[test]
fn test_bytes_revator() {
    let s = "ศไทย中华Việt Nam";
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
        184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
        109
    ];
    let mut pos = v.len();

    for b in s.bytes().rev() {
        pos -= 1;
        assert_eq!(b, v[pos]);
    }
}

#[test]
fn test_bytesator_nth() {
    let s = "ศไทย中华Việt Nam";
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
        184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
        109
    ];

    let mut b = s.bytes();
    assert_eq!(b.nth(2).unwrap(), v[2]);
    assert_eq!(b.nth(10).unwrap(), v[10]);
    assert_eq!(b.nth(200), None);
}

#[test]
fn test_bytesator_count() {
    let s = "ศไทย中华Việt Nam";

    let b = s.bytes();
    assert_eq!(b.count(), 28)
}

#[test]
fn test_bytesator_last() {
    let s = "ศไทย中华Việt Nam";

    let b = s.bytes();
    assert_eq!(b.last().unwrap(), 109)
}

#[test]
fn test_char_indicesator() {
    let s = "ศไทย中华Việt Nam";
    let p = [0, 3, 6, 9, 12, 15, 18, 19, 20, 23, 24, 25, 26, 27];
    let v = ['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];

    let mut pos = 0;
    let it = s.char_indices();

    for c in it {
        assert_eq!(c, (p[pos], v[pos]));
        pos += 1;
    }
    assert_eq!(pos, v.len());
    assert_eq!(pos, p.len());
}

#[test]
fn test_char_indices_revator() {
    let s = "ศไทย中华Việt Nam";
    let p = [27, 26, 25, 24, 23, 20, 19, 18, 15, 12, 9, 6, 3, 0];
    let v = ['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

    let mut pos = 0;
    let it = s.char_indices().rev();

    for c in it {
        assert_eq!(c, (p[pos], v[pos]));
        pos += 1;
    }
    assert_eq!(pos, v.len());
    assert_eq!(pos, p.len());
}

#[test]
fn test_splitn_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.splitn(4, ' ').collect();
    assert_eq!(split, ["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

    let split: Vec<&str> = data.splitn(4, |c: char| c == ' ').collect();
    assert_eq!(split, ["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

    // Unicode
    let split: Vec<&str> = data.splitn(4, 'ä').collect();
    assert_eq!(split, ["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);

    let split: Vec<&str> = data.splitn(4, |c: char| c == 'ä').collect();
    assert_eq!(split, ["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);
}

#[test]
fn test_split_char_iterator_no_trailing() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split('\n').collect();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb", ""]);

    let split: Vec<&str> = data.split_terminator('\n').collect();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb"]);
}

#[test]
fn test_rsplit() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.rsplit(' ').collect();
    assert_eq!(split, ["lämb\n", "lämb\nLittle", "little", "ä", "häd", "\nMäry"]);

    let split: Vec<&str> = data.rsplit("lämb").collect();
    assert_eq!(split, ["\n", "\nLittle ", "\nMäry häd ä little "]);

    let split: Vec<&str> = data.rsplit(|c: char| c == 'ä').collect();
    assert_eq!(split, ["mb\n", "mb\nLittle l", " little l", "d ", "ry h", "\nM"]);
}

#[test]
fn test_rsplitn() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.rsplitn(2, ' ').collect();
    assert_eq!(split, ["lämb\n", "\nMäry häd ä little lämb\nLittle"]);

    let split: Vec<&str> = data.rsplitn(2, "lämb").collect();
    assert_eq!(split, ["\n", "\nMäry häd ä little lämb\nLittle "]);

    let split: Vec<&str> = data.rsplitn(2, |c: char| c == 'ä').collect();
    assert_eq!(split, ["mb\n", "\nMäry häd ä little lämb\nLittle l"]);
}

#[test]
fn test_split_whitespace() {
    let data = "\n \tMäry   häd\tä  little lämb\nLittle lämb\n";
    let words: Vec<&str> = data.split_whitespace().collect();
    assert_eq!(words, ["Märy", "häd", "ä", "little", "lämb", "Little", "lämb"])
}

#[test]
fn test_lines() {
    let data = "\nMäry häd ä little lämb\n\r\nLittle lämb\n";
    let lines: Vec<&str> = data.lines().collect();
    assert_eq!(lines, ["", "Märy häd ä little lämb", "", "Little lämb"]);

    let data = "\r\nMäry häd ä little lämb\n\nLittle lämb"; // no trailing \n
    let lines: Vec<&str> = data.lines().collect();
    assert_eq!(lines, ["", "Märy häd ä little lämb", "", "Little lämb"]);
}

#[test]
fn test_splitator() {
    fn t(s: &str, sep: &str, u: &[&str]) {
        let v: Vec<&str> = s.split(sep).collect();
        assert_eq!(v, u);
    }
    t("--1233345--", "12345", &["--1233345--"]);
    t("abc::hello::there", "::", &["abc", "hello", "there"]);
    t("::hello::there", "::", &["", "hello", "there"]);
    t("hello::there::", "::", &["hello", "there", ""]);
    t("::hello::there::", "::", &["", "hello", "there", ""]);
    t("ประเทศไทย中华Việt Nam", "中华", &["ประเทศไทย", "Việt Nam"]);
    t("zzXXXzzYYYzz", "zz", &["", "XXX", "YYY", ""]);
    t("zzXXXzYYYz", "XXX", &["zz", "zYYYz"]);
    t(".XXX.YYY.", ".", &["", "XXX", "YYY", ""]);
    t("", ".", &[""]);
    t("zz", "zz", &["",""]);
    t("ok", "z", &["ok"]);
    t("zzz", "zz", &["","z"]);
    t("zzzzz", "zz", &["","","z"]);
}

#[test]
fn test_str_default() {
    use std::default::Default;

    fn t<S: Default + AsRef<str>>() {
        let s: S = Default::default();
        assert_eq!(s.as_ref(), "");
    }

    t::<&str>();
    t::<String>();
}

#[test]
fn test_str_container() {
    fn sum_len(v: &[&str]) -> usize {
        v.iter().map(|x| x.len()).sum()
    }

    let s = "01234";
    assert_eq!(5, sum_len(&["012", "", "34"]));
    assert_eq!(5, sum_len(&["01", "2", "34", ""]));
    assert_eq!(5, sum_len(&[s]));
}

#[test]
fn test_str_from_utf8() {
    let xs = b"hello";
    assert_eq!(from_utf8(xs), Ok("hello"));

    let xs = "ศไทย中华Việt Nam".as_bytes();
    assert_eq!(from_utf8(xs), Ok("ศไทย中华Việt Nam"));

    let xs = b"hello\xFF";
    assert!(from_utf8(xs).is_err());
}

#[test]
fn test_pattern_deref_forward() {
    let data = "aabcdaa";
    assert!(data.contains("bcd"));
    assert!(data.contains(&"bcd"));
    assert!(data.contains(&"bcd".to_string()));
}

#[test]
fn test_empty_match_indices() {
    let data = "aä中!";
    let vec: Vec<_> = data.match_indices("").collect();
    assert_eq!(vec, [(0, ""), (1, ""), (3, ""), (6, ""), (7, "")]);
}

#[test]
fn test_bool_from_str() {
    assert_eq!("true".parse().ok(), Some(true));
    assert_eq!("false".parse().ok(), Some(false));
    assert_eq!("not even a boolean".parse::<bool>().ok(), None);
}

fn check_contains_all_substrings(s: &str) {
    assert!(s.contains(""));
    for i in 0..s.len() {
        for j in i+1..s.len() + 1 {
            assert!(s.contains(&s[i..j]));
        }
    }
}

#[test]
fn strslice_issue_16589() {
    assert!("bananas".contains("nana"));

    // prior to the fix for #16589, x.contains("abcdabcd") returned false
    // test all substrings for good measure
    check_contains_all_substrings("012345678901234567890123456789bcdabcdabcd");
}

#[test]
fn strslice_issue_16878() {
    assert!(!"1234567ah012345678901ah".contains("hah"));
    assert!(!"00abc01234567890123456789abc".contains("bcabc"));
}


#[test]
fn test_strslice_contains() {
    let x = "There are moments, Jeeves, when one asks oneself, 'Do trousers matter?'";
    check_contains_all_substrings(x);
}

#[test]
fn test_rsplitn_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let mut split: Vec<&str> = data.rsplitn(4, ' ').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut split: Vec<&str> = data.rsplitn(4, |c: char| c == ' ').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let mut split: Vec<&str> = data.rsplitn(4, 'ä').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);

    let mut split: Vec<&str> = data.rsplitn(4, |c: char| c == 'ä').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);
}

#[test]
fn test_split_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split(' ').collect();
    assert_eq!( split, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == ' ').collect();
    assert_eq!( split, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(|c: char| c == ' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let split: Vec<&str> = data.split('ä').collect();
    assert_eq!( split, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let mut rsplit: Vec<&str> = data.split('ä').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == 'ä').collect();
    assert_eq!( split, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let mut rsplit: Vec<&str> = data.split(|c: char| c == 'ä').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);
}

#[test]
fn test_rev_split_char_iterator_no_trailing() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let mut split: Vec<&str> = data.split('\n').rev().collect();
    split.reverse();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb", ""]);

    let mut split: Vec<&str> = data.split_terminator('\n').rev().collect();
    split.reverse();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb"]);
}

#[test]
fn test_utf16_code_units() {
    use rustc_unicode::str::Utf16Encoder;
    assert_eq!(Utf16Encoder::new(vec!['é', '\u{1F4A9}'].into_iter()).collect::<Vec<u16>>(),
               [0xE9, 0xD83D, 0xDCA9])
}

#[test]
fn starts_with_in_unicode() {
    assert!(!"├── Cargo.toml".starts_with("# "));
}

#[test]
fn starts_short_long() {
    assert!(!"".starts_with("##"));
    assert!(!"##".starts_with("####"));
    assert!("####".starts_with("##"));
    assert!(!"##ä".starts_with("####"));
    assert!("####ä".starts_with("##"));
    assert!(!"##".starts_with("####ä"));
    assert!("##ä##".starts_with("##ä"));

    assert!("".starts_with(""));
    assert!("ä".starts_with(""));
    assert!("#ä".starts_with(""));
    assert!("##ä".starts_with(""));
    assert!("ä###".starts_with(""));
    assert!("#ä##".starts_with(""));
    assert!("##ä#".starts_with(""));
}

#[test]
fn contains_weird_cases() {
    assert!("* \t".contains(' '));
    assert!(!"* \t".contains('?'));
    assert!(!"* \t".contains('\u{1F4A9}'));
}

#[test]
fn trim_ws() {
    assert_eq!(" \t  a \t  ".trim_left_matches(|c: char| c.is_whitespace()),
                    "a \t  ");
    assert_eq!(" \t  a \t  ".trim_right_matches(|c: char| c.is_whitespace()),
               " \t  a");
    assert_eq!(" \t  a \t  ".trim_matches(|c: char| c.is_whitespace()),
                    "a");
    assert_eq!(" \t   \t  ".trim_left_matches(|c: char| c.is_whitespace()),
                         "");
    assert_eq!(" \t   \t  ".trim_right_matches(|c: char| c.is_whitespace()),
               "");
    assert_eq!(" \t   \t  ".trim_matches(|c: char| c.is_whitespace()),
               "");
}

#[test]
fn to_lowercase() {
    assert_eq!("".to_lowercase(), "");
    assert_eq!("AÉǅaé ".to_lowercase(), "aéǆaé ");

    // https://github.com/rust-lang/rust/issues/26035
    assert_eq!("ΑΣ".to_lowercase(), "ας");
    assert_eq!("Α'Σ".to_lowercase(), "α'ς");
    assert_eq!("Α''Σ".to_lowercase(), "α''ς");

    assert_eq!("ΑΣ Α".to_lowercase(), "ας α");
    assert_eq!("Α'Σ Α".to_lowercase(), "α'ς α");
    assert_eq!("Α''Σ Α".to_lowercase(), "α''ς α");

    assert_eq!("ΑΣ' Α".to_lowercase(), "ας' α");
    assert_eq!("ΑΣ'' Α".to_lowercase(), "ας'' α");

    assert_eq!("Α'Σ' Α".to_lowercase(), "α'ς' α");
    assert_eq!("Α''Σ'' Α".to_lowercase(), "α''ς'' α");

    assert_eq!("Α Σ".to_lowercase(), "α σ");
    assert_eq!("Α 'Σ".to_lowercase(), "α 'σ");
    assert_eq!("Α ''Σ".to_lowercase(), "α ''σ");

    assert_eq!("Σ".to_lowercase(), "σ");
    assert_eq!("'Σ".to_lowercase(), "'σ");
    assert_eq!("''Σ".to_lowercase(), "''σ");

    assert_eq!("ΑΣΑ".to_lowercase(), "ασα");
    assert_eq!("ΑΣ'Α".to_lowercase(), "ασ'α");
    assert_eq!("ΑΣ''Α".to_lowercase(), "ασ''α");
}

#[test]
fn to_uppercase() {
    assert_eq!("".to_uppercase(), "");
    assert_eq!("aéǅßﬁᾀ".to_uppercase(), "AÉǄSSFIἈΙ");
}

#[test]
fn test_into_string() {
    // The only way to acquire a Box<str> in the first place is through a String, so just
    // test that we can round-trip between Box<str> and String.
    let string = String::from("Some text goes here");
    assert_eq!(string.clone().into_boxed_str().into_string(), string);
}

#[test]
fn test_box_slice_clone() {
    let data = String::from("hello HELLO hello HELLO yes YES 5 中ä华!!!");
    let data2 = data.clone().into_boxed_str().clone().into_string();

    assert_eq!(data, data2);
}

mod pattern {
    use std::str::pattern::Pattern;
    use std::str::pattern::{Searcher, ReverseSearcher};
    use std::str::pattern::SearchStep::{self, Match, Reject, Done};

    macro_rules! make_test {
        ($name:ident, $p:expr, $h:expr, [$($e:expr,)*]) => {
            #[allow(unused_imports)]
            mod $name {
                use std::str::pattern::SearchStep::{Match, Reject};
                use super::{cmp_search_to_vec};
                #[test]
                fn fwd() {
                    cmp_search_to_vec(false, $p, $h, vec![$($e),*]);
                }
                #[test]
                fn bwd() {
                    cmp_search_to_vec(true, $p, $h, vec![$($e),*]);
                }
            }
        }
    }

    fn cmp_search_to_vec<'a, P: Pattern<'a>>(rev: bool, pat: P, haystack: &'a str,
                                             right: Vec<SearchStep>)
    where P::Searcher: ReverseSearcher<'a>
    {
        let mut searcher = pat.into_searcher(haystack);
        let mut v = vec![];
        loop {
            match if !rev {searcher.next()} else {searcher.next_back()} {
                Match(a, b) => v.push(Match(a, b)),
                Reject(a, b) => v.push(Reject(a, b)),
                Done => break,
            }
        }
        if rev {
            v.reverse();
        }

        let mut first_index = 0;
        let mut err = None;

        for (i, e) in right.iter().enumerate() {
            match *e {
                Match(a, b) | Reject(a, b)
                if a <= b && a == first_index => {
                    first_index = b;
                }
                _ => {
                    err = Some(i);
                    break;
                }
            }
        }

        if let Some(err) = err {
            panic!("Input skipped range at {}", err);
        }

        if first_index != haystack.len() {
            panic!("Did not cover whole input");
        }

        assert_eq!(v, right);
    }

    make_test!(str_searcher_ascii_haystack, "bb", "abbcbbd", [
        Reject(0, 1),
        Match (1, 3),
        Reject(3, 4),
        Match (4, 6),
        Reject(6, 7),
    ]);
    make_test!(str_searcher_ascii_haystack_seq, "bb", "abbcbbbbd", [
        Reject(0, 1),
        Match (1, 3),
        Reject(3, 4),
        Match (4, 6),
        Match (6, 8),
        Reject(8, 9),
    ]);
    make_test!(str_searcher_empty_needle_ascii_haystack, "", "abbcbbd", [
        Match (0, 0),
        Reject(0, 1),
        Match (1, 1),
        Reject(1, 2),
        Match (2, 2),
        Reject(2, 3),
        Match (3, 3),
        Reject(3, 4),
        Match (4, 4),
        Reject(4, 5),
        Match (5, 5),
        Reject(5, 6),
        Match (6, 6),
        Reject(6, 7),
        Match (7, 7),
    ]);
    make_test!(str_searcher_mulibyte_haystack, " ", "├──", [
        Reject(0, 3),
        Reject(3, 6),
        Reject(6, 9),
    ]);
    make_test!(str_searcher_empty_needle_mulibyte_haystack, "", "├──", [
        Match (0, 0),
        Reject(0, 3),
        Match (3, 3),
        Reject(3, 6),
        Match (6, 6),
        Reject(6, 9),
        Match (9, 9),
    ]);
    make_test!(str_searcher_empty_needle_empty_haystack, "", "", [
        Match(0, 0),
    ]);
    make_test!(str_searcher_nonempty_needle_empty_haystack, "├", "", [
    ]);
    make_test!(char_searcher_ascii_haystack, 'b', "abbcbbd", [
        Reject(0, 1),
        Match (1, 2),
        Match (2, 3),
        Reject(3, 4),
        Match (4, 5),
        Match (5, 6),
        Reject(6, 7),
    ]);
    make_test!(char_searcher_mulibyte_haystack, ' ', "├──", [
        Reject(0, 3),
        Reject(3, 6),
        Reject(6, 9),
    ]);
    make_test!(char_searcher_short_haystack, '\u{1F4A9}', "* \t", [
        Reject(0, 1),
        Reject(1, 2),
        Reject(2, 3),
    ]);

}

macro_rules! generate_iterator_test {
    {
        $name:ident {
            $(
                ($($arg:expr),*) -> [$($t:tt)*];
            )*
        }
        with $fwd:expr, $bwd:expr;
    } => {
        #[test]
        fn $name() {
            $(
                {
                    let res = vec![$($t)*];

                    let fwd_vec: Vec<_> = ($fwd)($($arg),*).collect();
                    assert_eq!(fwd_vec, res);

                    let mut bwd_vec: Vec<_> = ($bwd)($($arg),*).collect();
                    bwd_vec.reverse();
                    assert_eq!(bwd_vec, res);
                }
            )*
        }
    };
    {
        $name:ident {
            $(
                ($($arg:expr),*) -> [$($t:tt)*];
            )*
        }
        with $fwd:expr;
    } => {
        #[test]
        fn $name() {
            $(
                {
                    let res = vec![$($t)*];

                    let fwd_vec: Vec<_> = ($fwd)($($arg),*).collect();
                    assert_eq!(fwd_vec, res);
                }
            )*
        }
    }
}

generate_iterator_test! {
    double_ended_split {
        ("foo.bar.baz", '.') -> ["foo", "bar", "baz"];
        ("foo::bar::baz", "::") -> ["foo", "bar", "baz"];
    }
    with str::split, str::rsplit;
}

generate_iterator_test! {
    double_ended_split_terminator {
        ("foo;bar;baz;", ';') -> ["foo", "bar", "baz"];
    }
    with str::split_terminator, str::rsplit_terminator;
}

generate_iterator_test! {
    double_ended_matches {
        ("a1b2c3", char::is_numeric) -> ["1", "2", "3"];
    }
    with str::matches, str::rmatches;
}

generate_iterator_test! {
    double_ended_match_indices {
        ("a1b2c3", char::is_numeric) -> [(1, "1"), (3, "2"), (5, "3")];
    }
    with str::match_indices, str::rmatch_indices;
}

generate_iterator_test! {
    not_double_ended_splitn {
        ("foo::bar::baz", 2, "::") -> ["foo", "bar::baz"];
    }
    with str::splitn;
}

generate_iterator_test! {
    not_double_ended_rsplitn {
        ("foo::bar::baz", 2, "::") -> ["baz", "foo::bar"];
    }
    with str::rsplitn;
}

mod bench {
    use test::{Bencher, black_box};

    #[bench]
    fn char_iterator(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| s.chars().count());
    }

    #[bench]
    fn char_iterator_for(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| {
            for ch in s.chars() { black_box(ch); }
        });
    }

    #[bench]
    fn char_iterator_ascii(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb";

        b.iter(|| s.chars().count());
    }

    #[bench]
    fn char_iterator_rev(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| s.chars().rev().count());
    }

    #[bench]
    fn char_iterator_rev_for(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| {
            for ch in s.chars().rev() { black_box(ch); }
        });
    }

    #[bench]
    fn char_indicesator(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.chars().count();

        b.iter(|| assert_eq!(s.char_indices().count(), len));
    }

    #[bench]
    fn char_indicesator_rev(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.chars().count();

        b.iter(|| assert_eq!(s.char_indices().rev().count(), len));
    }

    #[bench]
    fn split_unicode_ascii(b: &mut Bencher) {
        let s = "ประเทศไทย中华Việt Namประเทศไทย中华Việt Nam";

        b.iter(|| assert_eq!(s.split('V').count(), 3));
    }

    #[bench]
    fn split_ascii(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();

        b.iter(|| assert_eq!(s.split(' ').count(), len));
    }

    #[bench]
    fn split_extern_fn(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();
        fn pred(c: char) -> bool { c == ' ' }

        b.iter(|| assert_eq!(s.split(pred).count(), len));
    }

    #[bench]
    fn split_closure(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();

        b.iter(|| assert_eq!(s.split(|c: char| c == ' ').count(), len));
    }

    #[bench]
    fn split_slice(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();

        let c: &[char] = &[' '];
        b.iter(|| assert_eq!(s.split(c).count(), len));
    }

    #[bench]
    fn bench_join(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let sep = "→";
        let v = vec![s, s, s, s, s, s, s, s, s, s];
        b.iter(|| {
            assert_eq!(v.join(sep).len(), s.len() * 10 + sep.len() * 9);
        })
    }

    #[bench]
    fn bench_contains_short_short(b: &mut Bencher) {
        let haystack = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let needle = "sit";

        b.iter(|| {
            assert!(haystack.contains(needle));
        })
    }

    #[bench]
    fn bench_contains_short_long(b: &mut Bencher) {
        let haystack = "\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem sit amet dolor \
ultricies condimentum. Praesent iaculis purus elit, ac malesuada quam malesuada in. Duis sed orci \
eros. Suspendisse sit amet magna mollis, mollis nunc luctus, imperdiet mi. Integer fringilla non \
sem ut lacinia. Fusce varius tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec \
tempus vel, gravida nec quam.

In est dui, tincidunt sed tempus interdum, adipiscing laoreet ante. Etiam tempor, tellus quis \
sagittis interdum, nulla purus mattis sem, quis auctor erat odio ac tellus. In nec nunc sit amet \
diam volutpat molestie at sed ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan \
lorem ac dignissim placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat. In vel \
eleifend felis. Sed suscipit nulla lorem, sed mollis est sollicitudin et. Nam fermentum egestas \
interdum. Curabitur ut nisi justo.

Sed sollicitudin ipsum tellus, ut condimentum leo eleifend nec. Cras ut velit ante. Phasellus nec \
mollis odio. Mauris molestie erat in arcu mattis, at aliquet dolor vehicula. Quisque malesuada \
lectus sit amet nisi pretium, a condimentum ipsum porta. Morbi at dapibus diam. Praesent egestas \
est sed risus elementum, eu rutrum metus ultrices. Etiam fermentum consectetur magna, id rutrum \
felis accumsan a. Aliquam ut pellentesque libero. Sed mi nulla, lobortis eu tortor id, suscipit \
ultricies neque. Morbi iaculis sit amet risus at iaculis. Praesent eget ligula quis turpis \
feugiat suscipit vel non arcu. Interdum et malesuada fames ac ante ipsum primis in faucibus. \
Aliquam sit amet placerat lorem.

Cras a lacus vel ante posuere elementum. Nunc est leo, bibendum ut facilisis vel, bibendum at \
mauris. Nullam adipiscing diam vel odio ornare, luctus adipiscing mi luctus. Nulla facilisi. \
Mauris adipiscing bibendum neque, quis adipiscing lectus tempus et. Sed feugiat erat et nisl \
lobortis pharetra. Donec vitae erat enim. Nullam sit amet felis et quam lacinia tincidunt. Aliquam \
suscipit dapibus urna. Sed volutpat urna in magna pulvinar volutpat. Phasellus nec tellus ac diam \
cursus accumsan.

Nam lectus enim, dapibus non nisi tempor, consectetur convallis massa. Maecenas eleifend dictum \
feugiat. Etiam quis mauris vel risus luctus mattis a a nunc. Nullam orci quam, imperdiet id \
vehicula in, porttitor ut nibh. Duis sagittis adipiscing nisl vitae congue. Donec mollis risus eu \
leo suscipit, varius porttitor nulla porta. Pellentesque ut sem nec nisi euismod vehicula. Nulla \
malesuada sollicitudin quam eu fermentum.";
        let needle = "english";

        b.iter(|| {
            assert!(!haystack.contains(needle));
        })
    }

    #[bench]
    fn bench_contains_bad_naive(b: &mut Bencher) {
        let haystack = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let needle = "aaaaaaaab";

        b.iter(|| {
            assert!(!haystack.contains(needle));
        })
    }

    #[bench]
    fn bench_contains_equal(b: &mut Bencher) {
        let haystack = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let needle = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";

        b.iter(|| {
            assert!(haystack.contains(needle));
        })
    }

    macro_rules! make_test_inner {
        ($s:ident, $code:expr, $name:ident, $str:expr) => {
            #[bench]
            fn $name(bencher: &mut Bencher) {
                let mut $s = $str;
                black_box(&mut $s);
                bencher.iter(|| $code);
            }
        }
    }

    macro_rules! make_test {
        ($name:ident, $s:ident, $code:expr) => {
            mod $name {
                use test::Bencher;
                use test::black_box;

                // Short strings: 65 bytes each
                make_test_inner!($s, $code, short_ascii,
                    "Mary had a little lamb, Little lamb Mary had a littl lamb, lamb!");
                make_test_inner!($s, $code, short_mixed,
                    "ศไทย中华Việt Nam; Mary had a little lamb, Little lam!");
                make_test_inner!($s, $code, short_pile_of_poo,
                    "💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩!");
                make_test_inner!($s, $code, long_lorem_ipsum,"\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem sit amet dolor \
ultricies condimentum. Praesent iaculis purus elit, ac malesuada quam malesuada in. Duis sed orci \
eros. Suspendisse sit amet magna mollis, mollis nunc luctus, imperdiet mi. Integer fringilla non \
sem ut lacinia. Fusce varius tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec \
tempus vel, gravida nec quam.

In est dui, tincidunt sed tempus interdum, adipiscing laoreet ante. Etiam tempor, tellus quis \
sagittis interdum, nulla purus mattis sem, quis auctor erat odio ac tellus. In nec nunc sit amet \
diam volutpat molestie at sed ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan \
lorem ac dignissim placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat. In vel \
eleifend felis. Sed suscipit nulla lorem, sed mollis est sollicitudin et. Nam fermentum egestas \
interdum. Curabitur ut nisi justo.

Sed sollicitudin ipsum tellus, ut condimentum leo eleifend nec. Cras ut velit ante. Phasellus nec \
mollis odio. Mauris molestie erat in arcu mattis, at aliquet dolor vehicula. Quisque malesuada \
lectus sit amet nisi pretium, a condimentum ipsum porta. Morbi at dapibus diam. Praesent egestas \
est sed risus elementum, eu rutrum metus ultrices. Etiam fermentum consectetur magna, id rutrum \
felis accumsan a. Aliquam ut pellentesque libero. Sed mi nulla, lobortis eu tortor id, suscipit \
ultricies neque. Morbi iaculis sit amet risus at iaculis. Praesent eget ligula quis turpis \
feugiat suscipit vel non arcu. Interdum et malesuada fames ac ante ipsum primis in faucibus. \
Aliquam sit amet placerat lorem.

Cras a lacus vel ante posuere elementum. Nunc est leo, bibendum ut facilisis vel, bibendum at \
mauris. Nullam adipiscing diam vel odio ornare, luctus adipiscing mi luctus. Nulla facilisi. \
Mauris adipiscing bibendum neque, quis adipiscing lectus tempus et. Sed feugiat erat et nisl \
lobortis pharetra. Donec vitae erat enim. Nullam sit amet felis et quam lacinia tincidunt. Aliquam \
suscipit dapibus urna. Sed volutpat urna in magna pulvinar volutpat. Phasellus nec tellus ac diam \
cursus accumsan.

Nam lectus enim, dapibus non nisi tempor, consectetur convallis massa. Maecenas eleifend dictum \
feugiat. Etiam quis mauris vel risus luctus mattis a a nunc. Nullam orci quam, imperdiet id \
vehicula in, porttitor ut nibh. Duis sagittis adipiscing nisl vitae congue. Donec mollis risus eu \
leo suscipit, varius porttitor nulla porta. Pellentesque ut sem nec nisi euismod vehicula. Nulla \
malesuada sollicitudin quam eu fermentum!");
            }
        }
    }

    make_test!(chars_count, s, s.chars().count());

    make_test!(contains_bang_str, s, s.contains("!"));
    make_test!(contains_bang_char, s, s.contains('!'));

    make_test!(match_indices_a_str, s, s.match_indices("a").count());

    make_test!(split_a_str, s, s.split("a").count());

    make_test!(trim_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_matches(|c: char| c.is_ascii())
    });
    make_test!(trim_left_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_left_matches(|c: char| c.is_ascii())
    });
    make_test!(trim_right_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_right_matches(|c: char| c.is_ascii())
    });

    make_test!(find_underscore_char, s, s.find('_'));
    make_test!(rfind_underscore_char, s, s.rfind('_'));
    make_test!(find_underscore_str, s, s.find("_"));

    make_test!(find_zzz_char, s, s.find('\u{1F4A4}'));
    make_test!(rfind_zzz_char, s, s.rfind('\u{1F4A4}'));
    make_test!(find_zzz_str, s, s.find("\u{1F4A4}"));

    make_test!(split_space_char, s, s.split(' ').count());
    make_test!(split_terminator_space_char, s, s.split_terminator(' ').count());

    make_test!(splitn_space_char, s, s.splitn(10, ' ').count());
    make_test!(rsplitn_space_char, s, s.rsplitn(10, ' ').count());

    make_test!(split_space_str, s, s.split(" ").count());
    make_test!(split_ad_str, s, s.split("ad").count());
}
