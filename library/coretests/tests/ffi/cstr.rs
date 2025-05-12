use core::ffi::CStr;

#[test]
fn compares_as_u8s() {
    let a: &CStr = c"Hello!"; // Starts with ascii
    let a_bytes: &[u8] = a.to_bytes();
    assert!((..0b1000_0000).contains(&a_bytes[0]));

    let b: &CStr = c"こんにちは！"; // Starts with non ascii
    let b_bytes: &[u8] = b.to_bytes();
    assert!((0b1000_0000..).contains(&b_bytes[0]));

    assert_eq!(Ord::cmp(a, b), Ord::cmp(a_bytes, b_bytes));
    assert_eq!(PartialOrd::partial_cmp(a, b), PartialOrd::partial_cmp(a_bytes, b_bytes));
}

#[test]
fn debug() {
    let s = c"abc\x01\x02\n\xE2\x80\xA6\xFF";
    assert_eq!(format!("{s:?}"), r#""abc\x01\x02\n\xe2\x80\xa6\xff""#);
}
