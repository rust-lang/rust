// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

trait Iterate<'a> {
    type Ty: Valid;
    fn iterate(self);
}
impl<'a, T> Iterate<'a> for T where T: Check {
    default type Ty = ();
    default fn iterate(self) {}
}

trait Check {}
impl<'a, T> Check for T where <T as Iterate<'a>>::Ty: Valid {}

trait Valid {}

fn main() {
    Iterate::iterate(0);
}
