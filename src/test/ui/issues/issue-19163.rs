// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-19163.rs

#[macro_use] extern crate issue_19163;

use std::io::Write;

fn main() {
    let mut v = vec![];
    mywrite!(&v, "Hello world");
 //~^ error: cannot borrow immutable borrowed content as mutable
}
