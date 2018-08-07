// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ffi;
use std::ops::Deref;

const SIZE: usize = 38;

/// Like SmallVec but for C strings.
#[derive(Clone)]
pub enum SmallCStr {
    OnStack {
        data: [u8; SIZE],
        len_with_nul: u8,
    },
    OnHeap {
        data: ffi::CString,
    }
}

impl SmallCStr {
    #[inline]
    pub fn new(s: &str) -> SmallCStr {
        if s.len() < SIZE {
            let mut data = [0; SIZE];
            data[.. s.len()].copy_from_slice(s.as_bytes());
            let len_with_nul = s.len() + 1;

            // Make sure once that this is a valid CStr
            if let Err(e) = ffi::CStr::from_bytes_with_nul(&data[.. len_with_nul]) {
                panic!("The string \"{}\" cannot be converted into a CStr: {}", s, e);
            }

            SmallCStr::OnStack {
                data,
                len_with_nul: len_with_nul as u8,
            }
        } else {
            SmallCStr::OnHeap {
                data: ffi::CString::new(s).unwrap()
            }
        }
    }

    #[inline]
    pub fn as_c_str(&self) -> &ffi::CStr {
        match *self {
            SmallCStr::OnStack { ref data, len_with_nul } => {
                unsafe {
                    let slice = &data[.. len_with_nul as usize];
                    ffi::CStr::from_bytes_with_nul_unchecked(slice)
                }
            }
            SmallCStr::OnHeap { ref data } => {
                data.as_c_str()
            }
        }
    }

    #[inline]
    pub fn len_with_nul(&self) -> usize {
        match *self {
            SmallCStr::OnStack { len_with_nul, .. } => {
                len_with_nul as usize
            }
            SmallCStr::OnHeap { ref data } => {
                data.as_bytes_with_nul().len()
            }
        }
    }
}

impl Deref for SmallCStr {
    type Target = ffi::CStr;

    fn deref(&self) -> &ffi::CStr {
        self.as_c_str()
    }
}


#[test]
fn short() {
    const TEXT: &str = "abcd";
    let reference = ffi::CString::new(TEXT.to_string()).unwrap();

    let scs = SmallCStr::new(TEXT);

    assert_eq!(scs.len_with_nul(), TEXT.len() + 1);
    assert_eq!(scs.as_c_str(), reference.as_c_str());
    assert!(if let SmallCStr::OnStack { .. } = scs { true } else { false });
}

#[test]
fn empty() {
    const TEXT: &str = "";
    let reference = ffi::CString::new(TEXT.to_string()).unwrap();

    let scs = SmallCStr::new(TEXT);

    assert_eq!(scs.len_with_nul(), TEXT.len() + 1);
    assert_eq!(scs.as_c_str(), reference.as_c_str());
    assert!(if let SmallCStr::OnStack { .. } = scs { true } else { false });
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
    assert!(if let SmallCStr::OnHeap { .. } = scs { true } else { false });
}

#[test]
#[should_panic]
fn internal_nul() {
    let _ = SmallCStr::new("abcd\0def");
}
