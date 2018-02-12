// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::Cow;
use std::path::PathBuf;
use std::ffi::{CStr, CString, OSStr, OSString};

#[test]
fn test_cow_from() {
    const MSG: &'static str = "Hello, World";
    let s = MSG.to_string();
    assert_eq!(Cow::from(&s), Cow::Borrowed(MSG));
    assert_eq!(Cow::from(s.as_str()), Cow::Borrowed(MSG));
    assert_eq!(Cow::from(s), Cow::Owned(MSG.to_string()));

    const VALUES: &'static [u8] = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let v = VALUES.iter().collect::<Vec<_>>();
    assert_eq!(Cow::from(&v), Cow::Borrowed(VALUES));
    assert_eq!(Cow::from(v.as_slice()), Cow::Borrowed(VALUES));
    assert_eq!(Cow::from(v), Cow::Owned(VALUES.iter().collect::<Vec<_>>()));

    let p = PathBuf::new();
    assert_eq!(Cow::from(&p), Cow::Borrowed(p.as_path()));
    assert_eq!(Cow::from(v.as_path()), Cow::Borrowed(p.as_path()));

    let cstring = CString::new(MSG);
    let cstr = {
        const MSG_NULL_TERMINATED: &'static str = "Hello, World\0";
        CStr::from_bytes_with_nul(MSG_NULL_TERMINATED).unwrap()
    };
    assert_eq(Cow::from(&cstring), Cow::Borrowed(cstr));
    assert_eq(Cow::from(cstring.as_c_str()), Cow::Borrowed(cstr));

    let s = OSString::from(MSG.into());
    assert_eq!(Cow::from(&s), Cow::Borrowed(OSStr::new(msg)));
    assert_eq!(Cow::from(s.as_os_str()), Cow::Borrowed(OSStr::new(msg)));
}
