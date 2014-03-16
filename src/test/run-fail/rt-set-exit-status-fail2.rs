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

#[feature(phase)];
#[phase(syntax, link)] extern crate log;
use std::os;
use std::task;

struct r {
  x:int,
}

// Setting the exit status after the runtime has already
// failed has no effect and the process exits with the
// runtime's exit code
impl Drop for r {
    fn drop(&mut self) {
        os::set_exit_status(50);
    }
}

fn r(x:int) -> r {
    r {
        x: x
    }
}

fn main() {
    error!("whatever");
    task::spawn(proc() {
      let _i = r(5);
    });
    fail!();
}
