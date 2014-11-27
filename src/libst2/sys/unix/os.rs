// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use libc::{c_int, c_char};
use prelude::*;
use io::IoResult;
use sys::fs::FileDesc;

use os::TMPBUF_SZ;

/// Returns the platform-specific value of errno
pub fn errno() -> int { unimplemented!() }

/// Get a detailed string description for the given error number
pub fn error_string(errno: i32) -> String { unimplemented!() }

pub unsafe fn pipe() -> IoResult<(FileDesc, FileDesc)> { unimplemented!() }
