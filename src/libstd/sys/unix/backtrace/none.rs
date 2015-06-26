// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// No backtraces support.

use io;
use io::prelude::*;
use result::Result::Err;

#[inline(always)]
pub fn write(_w: &mut Write) -> io::Result<()> {
    use io::ErrorKind;
    Err(io::Error::new(ErrorKind::Other,
                       "no library is available to print stack backtraces"))
}
