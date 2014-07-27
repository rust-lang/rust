// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that method lookup automatically both borrows vectors to slices
// and also references them to create the &self pointer


trait MyIter {
    fn test_imm(&self);
}

impl<'a> MyIter for &'a [int] {
    fn test_imm(&self) { assert_eq!(self[0], 1) }
}

impl<'a> MyIter for &'a str {
    fn test_imm(&self) { assert_eq!(*self, "test") }
}

pub fn main() {
    ([1]).test_imm();
    (vec!(1)).as_slice().test_imm();
    (&[1]).test_imm();
    ("test").test_imm();
    ("test").test_imm();

    // FIXME: Other types of mutable vecs don't currently exist

    // NB: We don't do this double autoreffing for &mut self because that would
    // allow creating a mutable pointer to a temporary, which would be a source
    // of confusion
}
