// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(convert)]

use std::ffi::{OsStr, OsString};

fn main() {
    // Valid UTF-8
    let vec1: Vec<u8> = b"t\xC3\xA9st".to_vec();
    let oso1: OsString = OsString::from_platform_bytes(vec1).unwrap();
    assert!(oso1.to_bytes() == Some(b"t\xC3\xA9st"));
    assert!(oso1.to_str() == Some("t\u{E9}st"));
    // Not UTF-8
    let vec2: Vec<u8> = b"t\xE9st".to_vec();
    let oso2: OsString = OsString::from_platform_bytes(vec2).unwrap();
    if cfg!(windows) {
        assert!(oso2.to_bytes() == None);
    } else {
        assert!(oso2.to_bytes() == Some(b"t\xE9st"));
    }
    assert_eq!(oso2.to_str(), None);

    // Valid UTF-8
    let by1: &[u8] = b"t\xC3\xA9st";
    let oss1: &OsStr = OsStr::from_platform_bytes(by1).unwrap();
    assert_eq!(oss1.to_bytes().unwrap().as_ptr(), by1.as_ptr());
    assert_eq!(oss1.to_str().unwrap().as_ptr(), by1.as_ptr());
    // Not UTF-8
    let by2: &[u8] = b"t\xE9st";
    let oss2: &OsStr = OsStr::from_platform_bytes(by2).unwrap();
    if cfg!(windows) {
        assert_eq!(oss2.to_bytes(), None);
    } else {
        assert_eq!(oss2.to_bytes().unwrap().as_ptr(), by2.as_ptr());
    }
    assert_eq!(oss2.to_str(), None);

    if cfg!(windows) {
        // FIXME: needs valid-windows-utf16-invalid-unicode test cases
    }
}
