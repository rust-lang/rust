// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue24687_lib.rs
// compile-flags:-g

extern crate issue24687_lib as d;

fn main() {
    // Create a d, which has a destructor whose body will be trans'ed
    // into the generated code here, and thus the local debuginfo will
    // need references into the original source locations from
    // `importer` above.
    let _d = d::D("Hi");
}
