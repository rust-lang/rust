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
use std::path::{Path, PathBuf};
use std::ffi::{CStr, CString, OsStr, OsString};

#[test]
fn test_cow_from() {
    const MSG: &'static str = "Hello, World";
    let s = MSG.to_string();
    assert_eq!(Cow::<str>::from(&s), Cow::Borrowed(MSG));
    assert_eq!(Cow::from(s.as_str()), Cow::Borrowed(MSG));
    assert_eq!(
        Cow::from(s),
        || -> Cow<str> { Cow::Owned(MSG.to_string()) }()
    );

    const VALUES: &[u8] = &[1u8, 2, 3, 4, 5, 6, 7, 8];
    let v = VALUES.iter().map(|b| *b).collect::<Vec<u8>>();
    assert_eq!(Cow::<[u8]>::from(&v), Cow::Borrowed(VALUES));
    assert_eq!(Cow::from(v.as_slice()), Cow::Borrowed(VALUES));
    assert_eq!(
        Cow::from(v),
        || -> Cow<[u8]> { Cow::Owned(VALUES.iter().map(|b| *b).collect::<Vec<u8>>() )}()
    );

    let p = PathBuf::new();
    assert_eq!(Cow::<Path>::from(&p), Cow::Borrowed(Path::new("")));
    assert_eq!(Cow::from(p.as_path()), Cow::Borrowed(Path::new("")));
    assert_eq!(
        Cow::from(p),
        || -> Cow<Path> { Cow::Owned(PathBuf::new()) }()
    );

    let cstring = CString::new(MSG).unwrap();
    let cstr = {
        const MSG_NULL_TERMINATED: &'static str = "Hello, World\0";
        CStr::from_bytes_with_nul(MSG_NULL_TERMINATED.as_bytes()).unwrap()
    };
    assert_eq!(Cow::<CStr>::from(&cstring), Cow::Borrowed(cstr));
    assert_eq!(Cow::from(cstring.as_c_str()), Cow::Borrowed(cstr));

    let s = OsString::from(MSG.to_string());
    assert_eq!(Cow::<OsString>::from(&s), Cow::Borrowed(OsStr::new(MSG)));
    assert_eq!(Cow::from(s.as_os_str()), Cow::Borrowed(OsStr::new(MSG)));
}

#[test]
fn test_generic_cow_from() {
    struct VecWrapper {
        _inner: Vec<i32>,
    }

    impl VecWrapper {
        fn new<'a, T: Into<Cow<'a, [i32]>>>(val: T) -> Self {
            VecWrapper {
                _inner: val.into().into_owned(),
            }
        } 
    }

    let ints = vec![0i32, 1, 2, 3, 4, 5];
    let _vw0 = VecWrapper::new(ints.as_slice());
    let _vw1 = VecWrapper::new(&ints);
    let _vw2 = VecWrapper::new(ints);
}
