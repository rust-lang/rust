// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #21422, which was related to failing to
// add inference constraints that the operands of a binary operator
// should outlive the binary operation itself.

pub struct P<'a> {
    _ptr: *const &'a u8,
}

impl <'a> PartialEq for P<'a> {
    fn eq(&self, other: &P<'a>) -> bool {
        (self as *const _) == (other as *const _)
    }
}

fn main() {}
