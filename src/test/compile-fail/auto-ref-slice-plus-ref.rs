// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

fn main() {

    // Testing that method lookup does not automatically borrow
    // vectors to slices then automatically create a &mut self
    // reference.  That would allow creating a mutable pointer to a
    // temporary, which would be a source of confusion

    let mut a = ~[0];
    a.test_mut(); //~ ERROR does not implement any method in scope named `test_mut`
}

trait MyIter {
    fn test_mut(&mut self);
}

impl<'a> MyIter for &'a [int] {
    fn test_mut(&mut self) { }
}
