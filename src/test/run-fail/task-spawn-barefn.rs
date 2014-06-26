// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:Ensure that the child task runs by failing

use std::task;

fn main() {
    // the purpose of this test is to make sure that task::spawn()
    // works when provided with a bare function:
    let r = task::try(startfn);
    if r.is_err() {
        fail!()
    }
}

fn startfn() {
    assert!("Ensure that the child task runs by failing".is_empty());
}
