// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Qux<'a> {
    s: &'a String
}

impl<'a> Qux<'a> {
    fn f(&self) {
        self.s.push('x');
        //~^ ERROR cannot borrow borrowed content `*self.s` of immutable binding as mutable
    }
}

fn main() {}
