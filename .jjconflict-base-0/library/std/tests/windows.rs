#![cfg(windows)]
//! An external tests

use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;
use std::path::PathBuf;

#[test]
#[should_panic]
fn os_string_must_know_it_isnt_utf8_issue_126291() {
    let mut utf8 = PathBuf::from(OsString::from("utf8".to_owned()));
    let non_utf8: OsString =
        OsStringExt::from_wide(&[0x6e, 0x6f, 0x6e, 0xd800, 0x75, 0x74, 0x66, 0x38]);
    utf8.set_extension(&non_utf8);
    utf8.into_os_string().into_string().unwrap();
}
