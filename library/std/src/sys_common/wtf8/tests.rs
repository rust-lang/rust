use super::*;
use crate::borrow::Cow;

#[test]
fn code_point_from_u32() {
    assert!(CodePoint::from_u32(0).is_some());
    assert!(CodePoint::from_u32(0xD800).is_some());
    assert!(CodePoint::from_u32(0x10FFFF).is_some());
    assert!(CodePoint::from_u32(0x110000).is_none());
}

#[test]
fn code_point_to_u32() {
    fn c(value: u32) -> CodePoint {
        CodePoint::from_u32(value).unwrap()
    }
    assert_eq!(c(0).to_u32(), 0);
    assert_eq!(c(0xD800).to_u32(), 0xD800);
    assert_eq!(c(0x10FFFF).to_u32(), 0x10FFFF);
}

#[test]
fn code_point_to_lead_surrogate() {
    fn c(value: u32) -> CodePoint {
        CodePoint::from_u32(value).unwrap()
    }
    assert_eq!(c(0).to_lead_surrogate(), None);
    assert_eq!(c(0xE9).to_lead_surrogate(), None);
    assert_eq!(c(0xD800).to_lead_surrogate(), Some(0xD800));
    assert_eq!(c(0xDBFF).to_lead_surrogate(), Some(0xDBFF));
    assert_eq!(c(0xDC00).to_lead_surrogate(), None);
    assert_eq!(c(0xDFFF).to_lead_surrogate(), None);
    assert_eq!(c(0x1F4A9).to_lead_surrogate(), None);
    assert_eq!(c(0x10FFFF).to_lead_surrogate(), None);
}

#[test]
fn code_point_to_trail_surrogate() {
    fn c(value: u32) -> CodePoint {
        CodePoint::from_u32(value).unwrap()
    }
    assert_eq!(c(0).to_trail_surrogate(), None);
    assert_eq!(c(0xE9).to_trail_surrogate(), None);
    assert_eq!(c(0xD800).to_trail_surrogate(), None);
    assert_eq!(c(0xDBFF).to_trail_surrogate(), None);
    assert_eq!(c(0xDC00).to_trail_surrogate(), Some(0xDC00));
    assert_eq!(c(0xDFFF).to_trail_surrogate(), Some(0xDFFF));
    assert_eq!(c(0x1F4A9).to_trail_surrogate(), None);
    assert_eq!(c(0x10FFFF).to_trail_surrogate(), None);
}

#[test]
fn code_point_from_char() {
    assert_eq!(CodePoint::from_char('a').to_u32(), 0x61);
    assert_eq!(CodePoint::from_char('💩').to_u32(), 0x1F4A9);
}

#[test]
fn code_point_to_string() {
    assert_eq!(format!("{:?}", CodePoint::from_char('a')), "U+0061");
    assert_eq!(format!("{:?}", CodePoint::from_char('💩')), "U+1F4A9");
}

#[test]
fn code_point_to_char() {
    fn c(value: u32) -> CodePoint {
        CodePoint::from_u32(value).unwrap()
    }
    assert_eq!(c(0x61).to_char(), Some('a'));
    assert_eq!(c(0x1F4A9).to_char(), Some('💩'));
    assert_eq!(c(0xD800).to_char(), None);
}

#[test]
fn code_point_to_char_lossy() {
    fn c(value: u32) -> CodePoint {
        CodePoint::from_u32(value).unwrap()
    }
    assert_eq!(c(0x61).to_char_lossy(), 'a');
    assert_eq!(c(0x1F4A9).to_char_lossy(), '💩');
    assert_eq!(c(0xD800).to_char_lossy(), '\u{FFFD}');
}

#[test]
fn wtf8buf_new() {
    assert_eq!(Wtf8Buf::new().bytes, b"");
}

#[test]
fn wtf8buf_from_str() {
    assert_eq!(Wtf8Buf::from_str("").bytes, b"");
    assert_eq!(Wtf8Buf::from_str("aé 💩").bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
}

#[test]
fn wtf8buf_from_string() {
    assert_eq!(Wtf8Buf::from_string(String::from("")).bytes, b"");
    assert_eq!(Wtf8Buf::from_string(String::from("aé 💩")).bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
}

#[test]
fn wtf8buf_from_wide() {
    let buf = Wtf8Buf::from_wide(&[]);
    assert_eq!(buf.bytes, b"");
    assert!(buf.is_known_utf8);

    let buf = Wtf8Buf::from_wide(&[0x61, 0xE9, 0x20, 0xD83D, 0xDCA9]);
    assert_eq!(buf.bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
    assert!(buf.is_known_utf8);

    let buf = Wtf8Buf::from_wide(&[0x61, 0xE9, 0x20, 0xD83D, 0xD83D, 0xDCA9]);
    assert_eq!(buf.bytes, b"a\xC3\xA9 \xED\xA0\xBD\xF0\x9F\x92\xA9");
    assert!(!buf.is_known_utf8);

    let buf = Wtf8Buf::from_wide(&[0xD800]);
    assert_eq!(buf.bytes, b"\xED\xA0\x80");
    assert!(!buf.is_known_utf8);

    let buf = Wtf8Buf::from_wide(&[0xDBFF]);
    assert_eq!(buf.bytes, b"\xED\xAF\xBF");
    assert!(!buf.is_known_utf8);

    let buf = Wtf8Buf::from_wide(&[0xDC00]);
    assert_eq!(buf.bytes, b"\xED\xB0\x80");
    assert!(!buf.is_known_utf8);

    let buf = Wtf8Buf::from_wide(&[0xDFFF]);
    assert_eq!(buf.bytes, b"\xED\xBF\xBF");
    assert!(!buf.is_known_utf8);
}

#[test]
fn wtf8buf_push_str() {
    let mut string = Wtf8Buf::new();
    assert_eq!(string.bytes, b"");
    assert!(string.is_known_utf8);

    string.push_str("aé 💩");
    assert_eq!(string.bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
    assert!(string.is_known_utf8);
}

#[test]
fn wtf8buf_push_char() {
    let mut string = Wtf8Buf::from_str("aé ");
    assert_eq!(string.bytes, b"a\xC3\xA9 ");
    assert!(string.is_known_utf8);

    string.push_char('💩');
    assert_eq!(string.bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
    assert!(string.is_known_utf8);
}

#[test]
fn wtf8buf_push() {
    let mut string = Wtf8Buf::from_str("aé ");
    assert_eq!(string.bytes, b"a\xC3\xA9 ");
    assert!(string.is_known_utf8);

    string.push(CodePoint::from_char('💩'));
    assert_eq!(string.bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
    assert!(string.is_known_utf8);

    fn c(value: u32) -> CodePoint {
        CodePoint::from_u32(value).unwrap()
    }

    let mut string = Wtf8Buf::new();
    string.push(c(0xD83D)); // lead
    assert!(!string.is_known_utf8);
    string.push(c(0xDCA9)); // trail
    assert_eq!(string.bytes, b"\xF0\x9F\x92\xA9"); // Magic!

    let mut string = Wtf8Buf::new();
    string.push(c(0xD83D)); // lead
    assert!(!string.is_known_utf8);
    string.push(c(0x20)); // not surrogate
    string.push(c(0xDCA9)); // trail
    assert_eq!(string.bytes, b"\xED\xA0\xBD \xED\xB2\xA9");

    let mut string = Wtf8Buf::new();
    string.push(c(0xD800)); // lead
    assert!(!string.is_known_utf8);
    string.push(c(0xDBFF)); // lead
    assert_eq!(string.bytes, b"\xED\xA0\x80\xED\xAF\xBF");

    let mut string = Wtf8Buf::new();
    string.push(c(0xD800)); // lead
    assert!(!string.is_known_utf8);
    string.push(c(0xE000)); // not surrogate
    assert_eq!(string.bytes, b"\xED\xA0\x80\xEE\x80\x80");

    let mut string = Wtf8Buf::new();
    string.push(c(0xD7FF)); // not surrogate
    assert!(string.is_known_utf8);
    string.push(c(0xDC00)); // trail
    assert!(!string.is_known_utf8);
    assert_eq!(string.bytes, b"\xED\x9F\xBF\xED\xB0\x80");

    let mut string = Wtf8Buf::new();
    string.push(c(0x61)); // not surrogate, < 3 bytes
    assert!(string.is_known_utf8);
    string.push(c(0xDC00)); // trail
    assert!(!string.is_known_utf8);
    assert_eq!(string.bytes, b"\x61\xED\xB0\x80");

    let mut string = Wtf8Buf::new();
    string.push(c(0xDC00)); // trail
    assert!(!string.is_known_utf8);
    assert_eq!(string.bytes, b"\xED\xB0\x80");
}

#[test]
fn wtf8buf_push_wtf8() {
    let mut string = Wtf8Buf::from_str("aé");
    assert_eq!(string.bytes, b"a\xC3\xA9");
    string.push_wtf8(Wtf8::from_str(" 💩"));
    assert_eq!(string.bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
    assert!(string.is_known_utf8);

    fn w(v: &[u8]) -> &Wtf8 {
        unsafe { Wtf8::from_bytes_unchecked(v) }
    }

    let mut string = Wtf8Buf::new();
    string.push_wtf8(w(b"\xED\xA0\xBD")); // lead
    string.push_wtf8(w(b"\xED\xB2\xA9")); // trail
    assert_eq!(string.bytes, b"\xF0\x9F\x92\xA9"); // Magic!

    let mut string = Wtf8Buf::new();
    string.push_wtf8(w(b"\xED\xA0\xBD")); // lead
    string.push_wtf8(w(b" ")); // not surrogate
    string.push_wtf8(w(b"\xED\xB2\xA9")); // trail
    assert_eq!(string.bytes, b"\xED\xA0\xBD \xED\xB2\xA9");
    assert!(!string.is_known_utf8);

    let mut string = Wtf8Buf::new();
    string.push_wtf8(w(b"\xED\xA0\x80")); // lead
    string.push_wtf8(w(b"\xED\xAF\xBF")); // lead
    assert_eq!(string.bytes, b"\xED\xA0\x80\xED\xAF\xBF");
    assert!(!string.is_known_utf8);

    let mut string = Wtf8Buf::new();
    string.push_wtf8(w(b"\xED\xA0\x80")); // lead
    string.push_wtf8(w(b"\xEE\x80\x80")); // not surrogate
    assert_eq!(string.bytes, b"\xED\xA0\x80\xEE\x80\x80");
    assert!(!string.is_known_utf8);

    let mut string = Wtf8Buf::new();
    string.push_wtf8(w(b"\xED\x9F\xBF")); // not surrogate
    string.push_wtf8(w(b"\xED\xB0\x80")); // trail
    assert_eq!(string.bytes, b"\xED\x9F\xBF\xED\xB0\x80");
    assert!(!string.is_known_utf8);

    let mut string = Wtf8Buf::new();
    string.push_wtf8(w(b"a")); // not surrogate, < 3 bytes
    string.push_wtf8(w(b"\xED\xB0\x80")); // trail
    assert_eq!(string.bytes, b"\x61\xED\xB0\x80");
    assert!(!string.is_known_utf8);

    let mut string = Wtf8Buf::new();
    string.push_wtf8(w(b"\xED\xB0\x80")); // trail
    assert_eq!(string.bytes, b"\xED\xB0\x80");
    assert!(!string.is_known_utf8);
}

#[test]
fn wtf8buf_truncate() {
    let mut string = Wtf8Buf::from_str("aé");
    assert!(string.is_known_utf8);

    string.truncate(3);
    assert_eq!(string.bytes, b"a\xC3\xA9");
    assert!(string.is_known_utf8);

    string.truncate(1);
    assert_eq!(string.bytes, b"a");
    assert!(string.is_known_utf8);

    string.truncate(0);
    assert_eq!(string.bytes, b"");
    assert!(string.is_known_utf8);
}

#[test]
fn wtf8buf_truncate_around_non_bmp() {
    let mut string = Wtf8Buf::from_str("💩");
    assert!(string.is_known_utf8);

    string.truncate(4);
    assert_eq!(string.bytes, b"\xF0\x9F\x92\xA9");
    assert!(string.is_known_utf8);

    string.truncate(0);
    assert_eq!(string.bytes, b"");
    assert!(string.is_known_utf8);
}

#[test]
#[should_panic]
fn wtf8buf_truncate_fail_code_point_boundary() {
    let mut string = Wtf8Buf::from_str("aé");
    string.truncate(2);
}

#[test]
#[should_panic]
fn wtf8buf_truncate_fail_longer() {
    let mut string = Wtf8Buf::from_str("aé");
    string.truncate(4);
}

#[test]
#[should_panic]
fn wtf8buf_truncate_splitting_non_bmp3() {
    let mut string = Wtf8Buf::from_str("💩");
    assert!(string.is_known_utf8);
    string.truncate(3);
}

#[test]
#[should_panic]
fn wtf8buf_truncate_splitting_non_bmp2() {
    let mut string = Wtf8Buf::from_str("💩");
    assert!(string.is_known_utf8);
    string.truncate(2);
}

#[test]
#[should_panic]
fn wtf8buf_truncate_splitting_non_bmp1() {
    let mut string = Wtf8Buf::from_str("💩");
    assert!(string.is_known_utf8);
    string.truncate(1);
}

#[test]
fn wtf8buf_into_string() {
    let mut string = Wtf8Buf::from_str("aé 💩");
    assert!(string.is_known_utf8);
    assert_eq!(string.clone().into_string(), Ok(String::from("aé 💩")));
    string.push(CodePoint::from_u32(0xD800).unwrap());
    assert!(!string.is_known_utf8);
    assert_eq!(string.clone().into_string(), Err(string));
}

#[test]
fn wtf8buf_into_string_lossy() {
    let mut string = Wtf8Buf::from_str("aé 💩");
    assert_eq!(string.clone().into_string_lossy(), String::from("aé 💩"));
    string.push(CodePoint::from_u32(0xD800).unwrap());
    assert_eq!(string.clone().into_string_lossy(), String::from("aé 💩�"));
}

#[test]
fn wtf8buf_from_iterator() {
    fn f(values: &[u32]) -> Wtf8Buf {
        values.iter().map(|&c| CodePoint::from_u32(c).unwrap()).collect::<Wtf8Buf>()
    }
    assert_eq!(
        f(&[0x61, 0xE9, 0x20, 0x1F4A9]),
        Wtf8Buf { bytes: b"a\xC3\xA9 \xF0\x9F\x92\xA9".to_vec(), is_known_utf8: true }
    );

    assert_eq!(f(&[0xD83D, 0xDCA9]).bytes, b"\xF0\x9F\x92\xA9"); // Magic!
    assert_eq!(
        f(&[0xD83D, 0x20, 0xDCA9]),
        Wtf8Buf { bytes: b"\xED\xA0\xBD \xED\xB2\xA9".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        f(&[0xD800, 0xDBFF]),
        Wtf8Buf { bytes: b"\xED\xA0\x80\xED\xAF\xBF".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        f(&[0xD800, 0xE000]),
        Wtf8Buf { bytes: b"\xED\xA0\x80\xEE\x80\x80".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        f(&[0xD7FF, 0xDC00]),
        Wtf8Buf { bytes: b"\xED\x9F\xBF\xED\xB0\x80".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        f(&[0x61, 0xDC00]),
        Wtf8Buf { bytes: b"\x61\xED\xB0\x80".to_vec(), is_known_utf8: false }
    );
    assert_eq!(f(&[0xDC00]), Wtf8Buf { bytes: b"\xED\xB0\x80".to_vec(), is_known_utf8: false });
}

#[test]
fn wtf8buf_extend() {
    fn e(initial: &[u32], extended: &[u32]) -> Wtf8Buf {
        fn c(value: &u32) -> CodePoint {
            CodePoint::from_u32(*value).unwrap()
        }
        let mut string = initial.iter().map(c).collect::<Wtf8Buf>();
        string.extend(extended.iter().map(c));
        string
    }

    assert_eq!(
        e(&[0x61, 0xE9], &[0x20, 0x1F4A9]),
        Wtf8Buf { bytes: b"a\xC3\xA9 \xF0\x9F\x92\xA9".to_vec(), is_known_utf8: true }
    );

    assert_eq!(e(&[0xD83D], &[0xDCA9]).bytes, b"\xF0\x9F\x92\xA9"); // Magic!
    assert_eq!(
        e(&[0xD83D, 0x20], &[0xDCA9]),
        Wtf8Buf { bytes: b"\xED\xA0\xBD \xED\xB2\xA9".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        e(&[0xD800], &[0xDBFF]),
        Wtf8Buf { bytes: b"\xED\xA0\x80\xED\xAF\xBF".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        e(&[0xD800], &[0xE000]),
        Wtf8Buf { bytes: b"\xED\xA0\x80\xEE\x80\x80".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        e(&[0xD7FF], &[0xDC00]),
        Wtf8Buf { bytes: b"\xED\x9F\xBF\xED\xB0\x80".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        e(&[0x61], &[0xDC00]),
        Wtf8Buf { bytes: b"\x61\xED\xB0\x80".to_vec(), is_known_utf8: false }
    );
    assert_eq!(
        e(&[], &[0xDC00]),
        Wtf8Buf { bytes: b"\xED\xB0\x80".to_vec(), is_known_utf8: false }
    );
}

#[test]
fn wtf8buf_show() {
    let mut string = Wtf8Buf::from_str("a\té \u{7f}💩\r");
    string.push(CodePoint::from_u32(0xD800).unwrap());
    assert_eq!(format!("{string:?}"), "\"a\\té \\u{7f}\u{1f4a9}\\r\\u{d800}\"");
}

#[test]
fn wtf8buf_as_slice() {
    assert_eq!(Wtf8Buf::from_str("aé").as_slice(), Wtf8::from_str("aé"));
}

#[test]
fn wtf8buf_show_str() {
    let text = "a\té 💩\r";
    let string = Wtf8Buf::from_str(text);
    assert_eq!(format!("{text:?}"), format!("{string:?}"));
}

#[test]
fn wtf8_from_str() {
    assert_eq!(&Wtf8::from_str("").bytes, b"");
    assert_eq!(&Wtf8::from_str("aé 💩").bytes, b"a\xC3\xA9 \xF0\x9F\x92\xA9");
}

#[test]
fn wtf8_len() {
    assert_eq!(Wtf8::from_str("").len(), 0);
    assert_eq!(Wtf8::from_str("aé 💩").len(), 8);
}

#[test]
fn wtf8_slice() {
    assert_eq!(&Wtf8::from_str("aé 💩")[1..4].bytes, b"\xC3\xA9 ");
}

#[test]
#[should_panic]
fn wtf8_slice_not_code_point_boundary() {
    let _ = &Wtf8::from_str("aé 💩")[2..4];
}

#[test]
fn wtf8_slice_from() {
    assert_eq!(&Wtf8::from_str("aé 💩")[1..].bytes, b"\xC3\xA9 \xF0\x9F\x92\xA9");
}

#[test]
#[should_panic]
fn wtf8_slice_from_not_code_point_boundary() {
    let _ = &Wtf8::from_str("aé 💩")[2..];
}

#[test]
fn wtf8_slice_to() {
    assert_eq!(&Wtf8::from_str("aé 💩")[..4].bytes, b"a\xC3\xA9 ");
}

#[test]
#[should_panic]
fn wtf8_slice_to_not_code_point_boundary() {
    let _ = &Wtf8::from_str("aé 💩")[5..];
}

#[test]
fn wtf8_ascii_byte_at() {
    let slice = Wtf8::from_str("aé 💩");
    assert_eq!(slice.ascii_byte_at(0), b'a');
    assert_eq!(slice.ascii_byte_at(1), b'\xFF');
    assert_eq!(slice.ascii_byte_at(2), b'\xFF');
    assert_eq!(slice.ascii_byte_at(3), b' ');
    assert_eq!(slice.ascii_byte_at(4), b'\xFF');
}

#[test]
fn wtf8_code_points() {
    fn c(value: u32) -> CodePoint {
        CodePoint::from_u32(value).unwrap()
    }
    fn cp(string: &Wtf8Buf) -> Vec<Option<char>> {
        string.code_points().map(|c| c.to_char()).collect::<Vec<_>>()
    }
    let mut string = Wtf8Buf::from_str("é ");
    assert_eq!(cp(&string), [Some('é'), Some(' ')]);
    string.push(c(0xD83D));
    assert_eq!(cp(&string), [Some('é'), Some(' '), None]);
    string.push(c(0xDCA9));
    assert_eq!(cp(&string), [Some('é'), Some(' '), Some('💩')]);
}

#[test]
fn wtf8_as_str() {
    assert_eq!(Wtf8::from_str("").as_str(), Some(""));
    assert_eq!(Wtf8::from_str("aé 💩").as_str(), Some("aé 💩"));
    let mut string = Wtf8Buf::new();
    string.push(CodePoint::from_u32(0xD800).unwrap());
    assert_eq!(string.as_str(), None);
}

#[test]
fn wtf8_to_string_lossy() {
    assert_eq!(Wtf8::from_str("").to_string_lossy(), Cow::Borrowed(""));
    assert_eq!(Wtf8::from_str("aé 💩").to_string_lossy(), Cow::Borrowed("aé 💩"));
    let mut string = Wtf8Buf::from_str("aé 💩");
    string.push(CodePoint::from_u32(0xD800).unwrap());
    let expected: Cow<'_, str> = Cow::Owned(String::from("aé 💩�"));
    assert_eq!(string.to_string_lossy(), expected);
}

#[test]
fn wtf8_display() {
    fn d(b: &[u8]) -> String {
        (&unsafe { Wtf8::from_bytes_unchecked(b) }).to_string()
    }

    assert_eq!("", d("".as_bytes()));
    assert_eq!("aé 💩", d("aé 💩".as_bytes()));

    let mut string = Wtf8Buf::from_str("aé 💩");
    string.push(CodePoint::from_u32(0xD800).unwrap());
    assert_eq!("aé 💩�", d(string.as_inner()));
}

#[test]
fn wtf8_encode_wide() {
    let mut string = Wtf8Buf::from_str("aé ");
    string.push(CodePoint::from_u32(0xD83D).unwrap());
    string.push_char('💩');
    assert_eq!(
        string.encode_wide().collect::<Vec<_>>(),
        vec![0x61, 0xE9, 0x20, 0xD83D, 0xD83D, 0xDCA9]
    );
}

#[test]
fn wtf8_encode_wide_size_hint() {
    let string = Wtf8Buf::from_str("\u{12345}");
    let mut iter = string.encode_wide();
    assert_eq!((1, Some(8)), iter.size_hint());
    iter.next().unwrap();
    assert_eq!((1, Some(1)), iter.size_hint());
    iter.next().unwrap();
    assert_eq!((0, Some(0)), iter.size_hint());
    assert!(iter.next().is_none());
}

#[test]
fn wtf8_clone_into() {
    let mut string = Wtf8Buf::new();
    Wtf8::from_str("green").clone_into(&mut string);
    assert_eq!(string.bytes, b"green");

    let mut string = Wtf8Buf::from_str("green");
    Wtf8::from_str("").clone_into(&mut string);
    assert_eq!(string.bytes, b"");

    let mut string = Wtf8Buf::from_str("red");
    Wtf8::from_str("green").clone_into(&mut string);
    assert_eq!(string.bytes, b"green");

    let mut string = Wtf8Buf::from_str("green");
    Wtf8::from_str("red").clone_into(&mut string);
    assert_eq!(string.bytes, b"red");

    let mut string = Wtf8Buf::from_str("green");
    assert!(string.is_known_utf8);
    unsafe { Wtf8::from_bytes_unchecked(b"\xED\xA0\x80").clone_into(&mut string) };
    assert_eq!(string.bytes, b"\xED\xA0\x80");
    assert!(!string.is_known_utf8);
}

#[test]
fn wtf8_to_ascii_lowercase() {
    let lowercase = Wtf8::from_str("").to_ascii_lowercase();
    assert_eq!(lowercase.bytes, b"");

    let lowercase = Wtf8::from_str("GrEeN gRaPeS! 🍇").to_ascii_lowercase();
    assert_eq!(lowercase.bytes, b"green grapes! \xf0\x9f\x8d\x87");

    let lowercase = unsafe { Wtf8::from_bytes_unchecked(b"\xED\xA0\x80").to_ascii_lowercase() };
    assert_eq!(lowercase.bytes, b"\xED\xA0\x80");
    assert!(!lowercase.is_known_utf8);
}

#[test]
fn wtf8_to_ascii_uppercase() {
    let uppercase = Wtf8::from_str("").to_ascii_uppercase();
    assert_eq!(uppercase.bytes, b"");

    let uppercase = Wtf8::from_str("GrEeN gRaPeS! 🍇").to_ascii_uppercase();
    assert_eq!(uppercase.bytes, b"GREEN GRAPES! \xf0\x9f\x8d\x87");

    let uppercase = unsafe { Wtf8::from_bytes_unchecked(b"\xED\xA0\x80").to_ascii_uppercase() };
    assert_eq!(uppercase.bytes, b"\xED\xA0\x80");
    assert!(!uppercase.is_known_utf8);
}

#[test]
fn wtf8_make_ascii_lowercase() {
    let mut lowercase = Wtf8Buf::from_str("");
    lowercase.make_ascii_lowercase();
    assert_eq!(lowercase.bytes, b"");

    let mut lowercase = Wtf8Buf::from_str("GrEeN gRaPeS! 🍇");
    lowercase.make_ascii_lowercase();
    assert_eq!(lowercase.bytes, b"green grapes! \xf0\x9f\x8d\x87");

    let mut lowercase = unsafe { Wtf8::from_bytes_unchecked(b"\xED\xA0\x80").to_owned() };
    lowercase.make_ascii_lowercase();
    assert_eq!(lowercase.bytes, b"\xED\xA0\x80");
    assert!(!lowercase.is_known_utf8);
}

#[test]
fn wtf8_make_ascii_uppercase() {
    let mut uppercase = Wtf8Buf::from_str("");
    uppercase.make_ascii_uppercase();
    assert_eq!(uppercase.bytes, b"");

    let mut uppercase = Wtf8Buf::from_str("GrEeN gRaPeS! 🍇");
    uppercase.make_ascii_uppercase();
    assert_eq!(uppercase.bytes, b"GREEN GRAPES! \xf0\x9f\x8d\x87");

    let mut uppercase = unsafe { Wtf8::from_bytes_unchecked(b"\xED\xA0\x80").to_owned() };
    uppercase.make_ascii_uppercase();
    assert_eq!(uppercase.bytes, b"\xED\xA0\x80");
    assert!(!uppercase.is_known_utf8);
}

#[test]
fn wtf8_to_owned() {
    let string = unsafe { Wtf8::from_bytes_unchecked(b"\xED\xA0\x80").to_owned() };
    assert_eq!(string.bytes, b"\xED\xA0\x80");
    assert!(!string.is_known_utf8);
}
