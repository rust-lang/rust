// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate extra;

use std::task;

pub fn main() { test00(); }

fn start(_task_number: int) { info!("Started / Finished task."); }

fn test00() {
    let i: int = 0;
    let mut builder = task::task();
    let mut result = builder.future_result();
    builder.spawn(proc() {
        start(i)
    });

    // Sleep long enough for the task to finish.
    let mut i = 0;
    while i < 10000 {
        task::deschedule();
        i += 1;
    }

    // Try joining tasks that have already finished.
    result.recv();

    info!("Joined task.");
}
