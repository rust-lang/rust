// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// copyright 2014 the rust project developers. see the copyright
// file at the top-level directory of this distribution and at
// http://rust-lang.org/copyright.
//
// licensed under the apache license, version 2.0 <license-apache or
// http://www.apache.org/licenses/license-2.0> or the mit license
// <license-mit or http://opensource.org/licenses/mit>, at your
// option. this file may not be copied, modified, or distributed
// except according to those terms.

// Test that cleanups for the RHS of shorcircuiting operators work.

use std::os;

pub fn main() {
    let args = os::args();

    // Here, the rvalue `"signal".to_owned()` requires cleanup. Older versions
    // of the code had a problem that the cleanup scope for this
    // expression was the end of the `if`, and as the `"signal".to_owned()`
    // expression was never evaluated, we wound up trying to clean
    // uninitialized memory.

    if args.len() >= 2 && args[1] == "signal".to_owned() {
        // Raise a segfault.
        unsafe { *(0 as *mut int) = 0; }
    }
}

