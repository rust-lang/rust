// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::Cow;
use std::iter::repeat;

use test::Bencher;

pub trait IntoCow<'a, B: ?Sized> where B: ToOwned {
    fn into_cow(self) -> Cow<'a, B>;
}

impl<'a> IntoCow<'a, str> for String {
    fn into_cow(self) -> Cow<'a, str> {
        Cow::Owned(self)
    }
}

impl<'a> IntoCow<'a, str> for &'a str {
    fn into_cow(self) -> Cow<'a, str> {
        Cow::Borrowed(self)
    }
}

#[test]
fn test_from_str() {
    let owned: Option<::std::string::String> = "string".parse().ok();
    assert_eq!(owned.as_ref().map(|s| &**s), Some("string"));
}

#[test]
fn test_from_cow_str() {
    assert_eq!(String::from(Cow::Borrowed("string")), "string");
    assert_eq!(String::from(Cow::Owned(String::from("string"))), "string");
}

#[test]
fn test_unsized_to_string() {
    let s: &str = "abc";
    let _: String = (*s).to_string();
}

#[test]
fn test_from_utf8() {
    let xs = b"hello".to_vec();
    assert_eq!(String::from_utf8(xs).unwrap(), String::from("hello"));

    let xs = "ศไทย中华Việt Nam".as_bytes().to_vec();
    assert_eq!(String::from_utf8(xs).unwrap(),
               String::from("ศไทย中华Việt Nam"));

    let xs = b"hello\xFF".to_vec();
    let err = String::from_utf8(xs).unwrap_err();
    assert_eq!(err.into_bytes(), b"hello\xff".to_vec());
}

#[test]
fn test_from_utf8_lossy() {
    let xs = b"hello";
    let ys: Cow<str> = "hello".into_cow();
    assert_eq!(String::from_utf8_lossy(xs), ys);

    let xs = "ศไทย中华Việt Nam".as_bytes();
    let ys: Cow<str> = "ศไทย中华Việt Nam".into_cow();
    assert_eq!(String::from_utf8_lossy(xs), ys);

    let xs = b"Hello\xC2 There\xFF Goodbye";
    assert_eq!(String::from_utf8_lossy(xs),
               String::from("Hello\u{FFFD} There\u{FFFD} Goodbye").into_cow());

    let xs = b"Hello\xC0\x80 There\xE6\x83 Goodbye";
    assert_eq!(String::from_utf8_lossy(xs),
               String::from("Hello\u{FFFD}\u{FFFD} There\u{FFFD} Goodbye").into_cow());

    let xs = b"\xF5foo\xF5\x80bar";
    assert_eq!(String::from_utf8_lossy(xs),
               String::from("\u{FFFD}foo\u{FFFD}\u{FFFD}bar").into_cow());

    let xs = b"\xF1foo\xF1\x80bar\xF1\x80\x80baz";
    assert_eq!(String::from_utf8_lossy(xs),
               String::from("\u{FFFD}foo\u{FFFD}bar\u{FFFD}baz").into_cow());

    let xs = b"\xF4foo\xF4\x80bar\xF4\xBFbaz";
    assert_eq!(String::from_utf8_lossy(xs),
               String::from("\u{FFFD}foo\u{FFFD}bar\u{FFFD}\u{FFFD}baz").into_cow());

    let xs = b"\xF0\x80\x80\x80foo\xF0\x90\x80\x80bar";
    assert_eq!(String::from_utf8_lossy(xs),
               String::from("\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}foo\u{10000}bar").into_cow());

    // surrogates
    let xs = b"\xED\xA0\x80foo\xED\xBF\xBFbar";
    assert_eq!(String::from_utf8_lossy(xs),
               String::from("\u{FFFD}\u{FFFD}\u{FFFD}foo\u{FFFD}\u{FFFD}\u{FFFD}bar").into_cow());
}

#[test]
fn test_from_utf16() {
    let pairs = [(String::from("𐍅𐌿𐌻𐍆𐌹𐌻𐌰\n"),
                  vec![0xd800, 0xdf45, 0xd800, 0xdf3f, 0xd800, 0xdf3b, 0xd800, 0xdf46, 0xd800,
                       0xdf39, 0xd800, 0xdf3b, 0xd800, 0xdf30, 0x000a]),

                 (String::from("𐐒𐑉𐐮𐑀𐐲𐑋 𐐏𐐲𐑍\n"),
                  vec![0xd801, 0xdc12, 0xd801, 0xdc49, 0xd801, 0xdc2e, 0xd801, 0xdc40, 0xd801,
                       0xdc32, 0xd801, 0xdc4b, 0x0020, 0xd801, 0xdc0f, 0xd801, 0xdc32, 0xd801,
                       0xdc4d, 0x000a]),

                 (String::from("𐌀𐌖𐌋𐌄𐌑𐌉·𐌌𐌄𐌕𐌄𐌋𐌉𐌑\n"),
                  vec![0xd800, 0xdf00, 0xd800, 0xdf16, 0xd800, 0xdf0b, 0xd800, 0xdf04, 0xd800,
                       0xdf11, 0xd800, 0xdf09, 0x00b7, 0xd800, 0xdf0c, 0xd800, 0xdf04, 0xd800,
                       0xdf15, 0xd800, 0xdf04, 0xd800, 0xdf0b, 0xd800, 0xdf09, 0xd800, 0xdf11,
                       0x000a]),

                 (String::from("𐒋𐒘𐒈𐒑𐒛𐒒 𐒕𐒓 𐒈𐒚𐒍 𐒏𐒜𐒒𐒖𐒆 𐒕𐒆\n"),
                  vec![0xd801, 0xdc8b, 0xd801, 0xdc98, 0xd801, 0xdc88, 0xd801, 0xdc91, 0xd801,
                       0xdc9b, 0xd801, 0xdc92, 0x0020, 0xd801, 0xdc95, 0xd801, 0xdc93, 0x0020,
                       0xd801, 0xdc88, 0xd801, 0xdc9a, 0xd801, 0xdc8d, 0x0020, 0xd801, 0xdc8f,
                       0xd801, 0xdc9c, 0xd801, 0xdc92, 0xd801, 0xdc96, 0xd801, 0xdc86, 0x0020,
                       0xd801, 0xdc95, 0xd801, 0xdc86, 0x000a]),
                 // Issue #12318, even-numbered non-BMP planes
                 (String::from("\u{20000}"), vec![0xD840, 0xDC00])];

    for p in &pairs {
        let (s, u) = (*p).clone();
        let s_as_utf16 = s.encode_utf16().collect::<Vec<u16>>();
        let u_as_string = String::from_utf16(&u).unwrap();

        assert!(::std_unicode::str::is_utf16(&u));
        assert_eq!(s_as_utf16, u);

        assert_eq!(u_as_string, s);
        assert_eq!(String::from_utf16_lossy(&u), s);

        assert_eq!(String::from_utf16(&s_as_utf16).unwrap(), s);
        assert_eq!(u_as_string.encode_utf16().collect::<Vec<u16>>(), u);
    }
}

#[test]
fn test_utf16_invalid() {
    // completely positive cases tested above.
    // lead + eof
    assert!(String::from_utf16(&[0xD800]).is_err());
    // lead + lead
    assert!(String::from_utf16(&[0xD800, 0xD800]).is_err());

    // isolated trail
    assert!(String::from_utf16(&[0x0061, 0xDC00]).is_err());

    // general
    assert!(String::from_utf16(&[0xD800, 0xd801, 0xdc8b, 0xD800]).is_err());
}

#[test]
fn test_from_utf16_lossy() {
    // completely positive cases tested above.
    // lead + eof
    assert_eq!(String::from_utf16_lossy(&[0xD800]),
               String::from("\u{FFFD}"));
    // lead + lead
    assert_eq!(String::from_utf16_lossy(&[0xD800, 0xD800]),
               String::from("\u{FFFD}\u{FFFD}"));

    // isolated trail
    assert_eq!(String::from_utf16_lossy(&[0x0061, 0xDC00]),
               String::from("a\u{FFFD}"));

    // general
    assert_eq!(String::from_utf16_lossy(&[0xD800, 0xd801, 0xdc8b, 0xD800]),
               String::from("\u{FFFD}𐒋\u{FFFD}"));
}

#[test]
fn test_push_bytes() {
    let mut s = String::from("ABC");
    unsafe {
        let mv = s.as_mut_vec();
        mv.extend_from_slice(&[b'D']);
    }
    assert_eq!(s, "ABCD");
}

#[test]
fn test_push_str() {
    let mut s = String::new();
    s.push_str("");
    assert_eq!(&s[0..], "");
    s.push_str("abc");
    assert_eq!(&s[0..], "abc");
    s.push_str("ประเทศไทย中华Việt Nam");
    assert_eq!(&s[0..], "abcประเทศไทย中华Việt Nam");
}

#[test]
fn test_add_assign() {
    let mut s = String::new();
    s += "";
    assert_eq!(s.as_str(), "");
    s += "abc";
    assert_eq!(s.as_str(), "abc");
    s += "ประเทศไทย中华Việt Nam";
    assert_eq!(s.as_str(), "abcประเทศไทย中华Việt Nam");
}

#[test]
fn test_push() {
    let mut data = String::from("ประเทศไทย中");
    data.push('华');
    data.push('b'); // 1 byte
    data.push('¢'); // 2 byte
    data.push('€'); // 3 byte
    data.push('𤭢'); // 4 byte
    assert_eq!(data, "ประเทศไทย中华b¢€𤭢");
}

#[test]
fn test_pop() {
    let mut data = String::from("ประเทศไทย中华b¢€𤭢");
    assert_eq!(data.pop().unwrap(), '𤭢'); // 4 bytes
    assert_eq!(data.pop().unwrap(), '€'); // 3 bytes
    assert_eq!(data.pop().unwrap(), '¢'); // 2 bytes
    assert_eq!(data.pop().unwrap(), 'b'); // 1 bytes
    assert_eq!(data.pop().unwrap(), '华');
    assert_eq!(data, "ประเทศไทย中");
}

#[test]
fn test_split_off_empty() {
    let orig = "Hello, world!";
    let mut split = String::from(orig);
    let empty: String = split.split_off(orig.len());
    assert!(empty.is_empty());
}

#[test]
#[should_panic]
fn test_split_off_past_end() {
    let orig = "Hello, world!";
    let mut split = String::from(orig);
    split.split_off(orig.len() + 1);
}

#[test]
#[should_panic]
fn test_split_off_mid_char() {
    let mut orig = String::from("山");
    orig.split_off(1);
}

#[test]
fn test_split_off_ascii() {
    let mut ab = String::from("ABCD");
    let cd = ab.split_off(2);
    assert_eq!(ab, "AB");
    assert_eq!(cd, "CD");
}

#[test]
fn test_split_off_unicode() {
    let mut nihon = String::from("日本語");
    let go = nihon.split_off("日本".len());
    assert_eq!(nihon, "日本");
    assert_eq!(go, "語");
}

#[test]
fn test_str_truncate() {
    let mut s = String::from("12345");
    s.truncate(5);
    assert_eq!(s, "12345");
    s.truncate(3);
    assert_eq!(s, "123");
    s.truncate(0);
    assert_eq!(s, "");

    let mut s = String::from("12345");
    let p = s.as_ptr();
    s.truncate(3);
    s.push_str("6");
    let p_ = s.as_ptr();
    assert_eq!(p_, p);
}

#[test]
fn test_str_truncate_invalid_len() {
    let mut s = String::from("12345");
    s.truncate(6);
    assert_eq!(s, "12345");
}

#[test]
#[should_panic]
fn test_str_truncate_split_codepoint() {
    let mut s = String::from("\u{FC}"); // ü
    s.truncate(1);
}

#[test]
fn test_str_clear() {
    let mut s = String::from("12345");
    s.clear();
    assert_eq!(s.len(), 0);
    assert_eq!(s, "");
}

#[test]
fn test_str_add() {
    let a = String::from("12345");
    let b = a + "2";
    let b = b + "2";
    assert_eq!(b.len(), 7);
    assert_eq!(b, "1234522");
}

#[test]
fn remove() {
    let mut s = "ศไทย中华Việt Nam; foobar".to_string();
    assert_eq!(s.remove(0), 'ศ');
    assert_eq!(s.len(), 33);
    assert_eq!(s, "ไทย中华Việt Nam; foobar");
    assert_eq!(s.remove(17), 'ệ');
    assert_eq!(s, "ไทย中华Vit Nam; foobar");
}

#[test]
#[should_panic]
fn remove_bad() {
    "ศ".to_string().remove(1);
}

#[test]
fn insert() {
    let mut s = "foobar".to_string();
    s.insert(0, 'ệ');
    assert_eq!(s, "ệfoobar");
    s.insert(6, 'ย');
    assert_eq!(s, "ệfooยbar");
}

#[test]
#[should_panic]
fn insert_bad1() {
    "".to_string().insert(1, 't');
}
#[test]
#[should_panic]
fn insert_bad2() {
    "ệ".to_string().insert(1, 't');
}

#[test]
fn test_slicing() {
    let s = "foobar".to_string();
    assert_eq!("foobar", &s[..]);
    assert_eq!("foo", &s[..3]);
    assert_eq!("bar", &s[3..]);
    assert_eq!("oob", &s[1..4]);
}

#[test]
fn test_simple_types() {
    assert_eq!(1.to_string(), "1");
    assert_eq!((-1).to_string(), "-1");
    assert_eq!(200.to_string(), "200");
    assert_eq!(2.to_string(), "2");
    assert_eq!(true.to_string(), "true");
    assert_eq!(false.to_string(), "false");
    assert_eq!(("hi".to_string()).to_string(), "hi");
}

#[test]
fn test_vectors() {
    let x: Vec<i32> = vec![];
    assert_eq!(format!("{:?}", x), "[]");
    assert_eq!(format!("{:?}", vec![1]), "[1]");
    assert_eq!(format!("{:?}", vec![1, 2, 3]), "[1, 2, 3]");
    assert!(format!("{:?}", vec![vec![], vec![1], vec![1, 1]]) == "[[], [1], [1, 1]]");
}

#[test]
fn test_from_iterator() {
    let s = "ศไทย中华Việt Nam".to_string();
    let t = "ศไทย中华";
    let u = "Việt Nam";

    let a: String = s.chars().collect();
    assert_eq!(s, a);

    let mut b = t.to_string();
    b.extend(u.chars());
    assert_eq!(s, b);

    let c: String = vec![t, u].into_iter().collect();
    assert_eq!(s, c);

    let mut d = t.to_string();
    d.extend(vec![u]);
    assert_eq!(s, d);
}

#[test]
fn test_drain() {
    let mut s = String::from("αβγ");
    assert_eq!(s.drain(2..4).collect::<String>(), "β");
    assert_eq!(s, "αγ");

    let mut t = String::from("abcd");
    t.drain(..0);
    assert_eq!(t, "abcd");
    t.drain(..1);
    assert_eq!(t, "bcd");
    t.drain(3..);
    assert_eq!(t, "bcd");
    t.drain(..);
    assert_eq!(t, "");
}

#[test]
fn test_extend_ref() {
    let mut a = "foo".to_string();
    a.extend(&['b', 'a', 'r']);

    assert_eq!(&a, "foobar");
}

#[test]
fn test_into_boxed_str() {
    let xs = String::from("hello my name is bob");
    let ys = xs.into_boxed_str();
    assert_eq!(&*ys, "hello my name is bob");
}

#[bench]
fn bench_with_capacity(b: &mut Bencher) {
    b.iter(|| String::with_capacity(100));
}

#[bench]
fn bench_push_str(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
    b.iter(|| {
        let mut r = String::new();
        r.push_str(s);
    });
}

const REPETITIONS: u64 = 10_000;

#[bench]
fn bench_push_str_one_byte(b: &mut Bencher) {
    b.bytes = REPETITIONS;
    b.iter(|| {
        let mut r = String::new();
        for _ in 0..REPETITIONS {
            r.push_str("a")
        }
    });
}

#[bench]
fn bench_push_char_one_byte(b: &mut Bencher) {
    b.bytes = REPETITIONS;
    b.iter(|| {
        let mut r = String::new();
        for _ in 0..REPETITIONS {
            r.push('a')
        }
    });
}

#[bench]
fn bench_push_char_two_bytes(b: &mut Bencher) {
    b.bytes = REPETITIONS * 2;
    b.iter(|| {
        let mut r = String::new();
        for _ in 0..REPETITIONS {
            r.push('â')
        }
    });
}

#[bench]
fn from_utf8_lossy_100_ascii(b: &mut Bencher) {
    let s = b"Hello there, the quick brown fox jumped over the lazy dog! \
              Lorem ipsum dolor sit amet, consectetur. ";

    assert_eq!(100, s.len());
    b.iter(|| {
        let _ = String::from_utf8_lossy(s);
    });
}

#[bench]
fn from_utf8_lossy_100_multibyte(b: &mut Bencher) {
    let s = "𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰".as_bytes();
    assert_eq!(100, s.len());
    b.iter(|| {
        let _ = String::from_utf8_lossy(s);
    });
}

#[bench]
fn from_utf8_lossy_invalid(b: &mut Bencher) {
    let s = b"Hello\xC0\x80 There\xE6\x83 Goodbye";
    b.iter(|| {
        let _ = String::from_utf8_lossy(s);
    });
}

#[bench]
fn from_utf8_lossy_100_invalid(b: &mut Bencher) {
    let s = repeat(0xf5).take(100).collect::<Vec<_>>();
    b.iter(|| {
        let _ = String::from_utf8_lossy(&s);
    });
}

#[bench]
fn bench_exact_size_shrink_to_fit(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    // ensure our operation produces an exact-size string before we benchmark it
    let mut r = String::with_capacity(s.len());
    r.push_str(s);
    assert_eq!(r.len(), r.capacity());
    b.iter(|| {
        let mut r = String::with_capacity(s.len());
        r.push_str(s);
        r.shrink_to_fit();
        r
    });
}

#[bench]
fn bench_from_str(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    b.iter(|| String::from(s))
}

#[bench]
fn bench_from(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    b.iter(|| String::from(s))
}

#[bench]
fn bench_to_string(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    b.iter(|| s.to_string())
}
