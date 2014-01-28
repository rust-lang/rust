// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A(~uint);

impl A {
    fn a(&self@&A(ref mut u), i: uint) {
        //~^ ERROR: cannot borrow immutable anonymous field as mutable
        **u = i;
    }

    fn b(self@A(u)) -> ~uint {
        let A(u) = self; // FIXME: Remove this line when #12534 is fixed
        //~^ NOTE: `self#0` moved here
        let _ = self;
        //~^ ERROR: use of partially moved value: `self`
        u
    }
}

fn main() {}
