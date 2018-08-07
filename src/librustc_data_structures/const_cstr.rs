// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// This macro creates a zero-overhead &CStr by adding a NUL terminator to
/// the string literal passed into it at compile-time. Use it like:
///
/// ```
///     let some_const_cstr = const_cstr!("abc");
/// ```
///
/// The above is roughly equivalent to:
///
/// ```
///     let some_const_cstr = CStr::from_bytes_with_nul(b"abc\0").unwrap()
/// ```
///
/// Note that macro only checks the string literal for internal NULs if
/// debug-assertions are enabled in order to avoid runtime overhead in release
/// builds.
#[macro_export]
macro_rules! const_cstr {
    ($s:expr) => ({
        use std::ffi::CStr;

        let str_plus_nul = concat!($s, "\0");

        if cfg!(debug_assertions) {
            CStr::from_bytes_with_nul(str_plus_nul.as_bytes()).unwrap()
        } else {
            unsafe {
                CStr::from_bytes_with_nul_unchecked(str_plus_nul.as_bytes())
            }
        }
    })
}
