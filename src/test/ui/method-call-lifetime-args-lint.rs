// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(late_bound_lifetime_arguments)]
#![allow(unused)]

struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    fn late_implicit(self, _: &u8, _: &u8) {}
}

fn method_call() {
    S.late::<'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted

    S.late_implicit::<'static>(&0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    //~| WARN this was previously accepted
}

fn main() {}
