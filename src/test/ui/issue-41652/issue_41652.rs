// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue_41652_b.rs

extern crate issue_41652_b;

struct S;

impl issue_41652_b::Tr for S {
    fn f() {
        3.f()
        //~^ ERROR can't call method `f` on ambiguous numeric type `{integer}`
    }
}

fn main() {}
