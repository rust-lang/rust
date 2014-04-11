// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/// Verify that mutable static items that are Share can be addressed without
/// an `unsafe` block.

struct Sharable {
    field: uint
}

impl Sharable {
    fn foo(&self) {}
    fn foo_mut(&mut self) {}
}

static mut MySharable: Sharable = Sharable { field: 1};

fn borrow_static(_: &Sharable) {}

pub fn main() {

    MySharable.foo();

    borrow_static(&MySharable);

    let _ = &MySharable;

    unsafe { let _: &mut Sharable = &mut MySharable; }

    let mut slc = ['a', 'c'];
    slc[MySharable.field] = 'b';

    let _ =  &((((MySharable))));
}
