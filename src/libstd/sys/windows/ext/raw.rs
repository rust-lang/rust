// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows-specific primitives

#![stable(feature = "raw_ext", since = "1.1.0")]

use os::raw::c_void;

#[stable(feature = "raw_ext", since = "1.1.0")] pub type HANDLE = *mut c_void;
#[cfg(target_pointer_width = "32")]
#[stable(feature = "raw_ext", since = "1.1.0")] pub type SOCKET = u32;
#[cfg(target_pointer_width = "64")]
#[stable(feature = "raw_ext", since = "1.1.0")] pub type SOCKET = u64;
