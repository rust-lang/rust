// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities related to FFI bindings.

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::c_str::{CString, CStr, NulError};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::os_str::{OsString, OsStr};

mod c_str;
mod os_str;

// FIXME (#21670): these should be defined in the os_str module
/// Freely convertible to an `&OsStr` slice.
#[unstable(feature = "std_misc")]
pub trait AsOsStr {
    /// Converts to an `&OsStr` slice.
    fn as_os_str(&self) -> &OsStr;
}
