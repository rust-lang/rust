// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

struct Struct;

impl Struct {
    fn bar(self: &mut Self) {
        //~^ WARN function cannot return without recursing
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
    }

    fn imm(self) {
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
    }

    fn mtbl(mut self) {
        (&mut self).bar();
    }

    fn immref(&self) {
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
        //~^^ ERROR cannot borrow data in a `&` reference as mutable [E0596]
    }

    fn mtblref(&mut self) {
        (&mut self).bar();
        //~^ ERROR cannot borrow `self` as mutable, as it is not declared as mutable [E0596]
    }
}

fn main () {}
