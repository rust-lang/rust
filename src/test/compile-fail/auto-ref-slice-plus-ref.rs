// Copyright 2012-14 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn main() {

    // Testing that method lookup does not automatically borrow
    // vectors to slices then automatically create a self reference.

    let mut a = vec![0];
    a.test_mut(); //~ ERROR no method named `test_mut` found
    a.test(); //~ ERROR no method named `test` found

    ([1]).test(); //~ ERROR no method named `test` found
    (&[1]).test(); //~ ERROR no method named `test` found
}

trait MyIter {
    fn test_mut(&mut self);
    fn test(&self);
}

impl<'a> MyIter for &'a [isize] {
    fn test_mut(&mut self) { }
    fn test(&self) { }
}

impl<'a> MyIter for &'a str {
    fn test_mut(&mut self) { }
    fn test(&self) { }
}
