// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(std_misc)]

use std::thread;

pub fn main() { test00(); }

fn start(_task_number: isize) { println!("Started / Finished task."); }

fn test00() {
    let i: isize = 0;
    let mut result = thread::spawn(move|| {
        start(i)
    });

    // Sleep long enough for the task to finish.
    let mut i = 0_usize;
    while i < 10000 {
        thread::yield_now();
        i += 1;
    }

    // Try joining tasks that have already finished.
    result.join();

    println!("Joined task.");
}
