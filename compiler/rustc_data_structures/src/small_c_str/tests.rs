use super::*;

#[test]
fn short() {
    const TEXT: &str = "abcd";
    let reference = ffi::CString::new(TEXT.to_string()).unwrap();

    let scs = SmallCStr::new(TEXT);

    assert_eq!(scs.len_with_nul(), TEXT.len() + 1);
    assert_eq!(scs.as_c_str(), reference.as_c_str());
    assert!(!scs.spilled());
}

#[test]
fn empty() {
    const TEXT: &str = "";
    let reference = ffi::CString::new(TEXT.to_string()).unwrap();

    let scs = SmallCStr::new(TEXT);

    assert_eq!(scs.len_with_nul(), TEXT.len() + 1);
    assert_eq!(scs.as_c_str(), reference.as_c_str());
    assert!(!scs.spilled());
}

#[test]
fn long() {
    const TEXT: &str = "01234567890123456789012345678901234567890123456789\
                        01234567890123456789012345678901234567890123456789\
                        01234567890123456789012345678901234567890123456789";
    let reference = ffi::CString::new(TEXT.to_string()).unwrap();

    let scs = SmallCStr::new(TEXT);

    assert_eq!(scs.len_with_nul(), TEXT.len() + 1);
    assert_eq!(scs.as_c_str(), reference.as_c_str());
    assert!(scs.spilled());
}

#[test]
#[should_panic]
fn internal_nul() {
    let _ = SmallCStr::new("abcd\0def");
}

#[test]
fn from_cstr() {
    let c = c"foo";
    let s: SmallCStr = c.into();
    assert_eq!(s.len_with_nul(), 4);
    assert_eq!(s.as_c_str(), c"foo");
}
