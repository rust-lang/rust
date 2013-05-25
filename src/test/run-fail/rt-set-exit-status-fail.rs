// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:whatever

use std::os;

fn main() {
    error!(~"whatever");
    // Setting the exit status only works when the scheduler terminates
    // normally. In this case we're going to fail, so instead of of
    // returning 50 the process will return the typical rt failure code.
    os::set_exit_status(50);
    fail!();
}
