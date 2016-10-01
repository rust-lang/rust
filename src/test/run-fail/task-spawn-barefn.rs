// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:Ensure that the child thread runs by panicking
// ignore-emscripten Needs threads.

use std::thread;

fn main() {
    // the purpose of this test is to make sure that thread::spawn()
    // works when provided with a bare function:
    let r = thread::spawn(startfn).join();
    if r.is_err() {
        panic!()
    }
}

fn startfn() {
    assert!("Ensure that the child thread runs by panicking".is_empty());
}
