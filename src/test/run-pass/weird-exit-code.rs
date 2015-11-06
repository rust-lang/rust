// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// On Windows the GetExitCodeProcess API is used to get the exit code of a
// process, but it's easy to mistake a process exiting with the code 259 as
// "still running" because this is the value of the STILL_ACTIVE constant. Make
// sure we handle this case in the standard library and correctly report the
// status.
//
// Note that this is disabled on unix as processes exiting with 259 will have
// their exit status truncated to 3 (only the lower 8 bits are used).

use std::process::{self, Command};
use std::env;

fn main() {
    if !cfg!(windows) {
        return
    }

    if env::args().len() == 1 {
        let status = Command::new(env::current_exe().unwrap())
                             .arg("foo")
                             .status()
                             .unwrap();
        assert_eq!(status.code(), Some(259));
    } else {
        process::exit(259);
    }
}
