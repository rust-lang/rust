#![feature(os_str_internals)]

use core::ffi::wtf8::*;

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
#[should_panic(expected = "byte index 4 is out of bounds")]
fn wtf8_utf8_boundary_out_of_bounds() {
    let string = Wtf8::from_str("aé");
    check_utf8_boundary(&string, 4);
}

#[test]
#[should_panic(expected = "byte index 1 is not a codepoint boundary")]
fn wtf8_utf8_boundary_inside_codepoint() {
    let string = Wtf8::from_str("é");
    check_utf8_boundary(&string, 1);
}
