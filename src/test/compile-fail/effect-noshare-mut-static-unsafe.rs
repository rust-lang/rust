// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/// Verify that mutable static items that are not Share can't be addressed without
/// an `unsafe` block.

use std::kinds::marker;

struct NonSharable {
    field: uint,
    noshare: marker::NoShare
}

struct Sharable {
    field: uint
}

impl Sharable {
    fn foo(&self) {}
    fn foo_mut(&mut self) {}
}

static mut MyNonSharable: NonSharable = NonSharable { field: 1, noshare: marker::NoShare };

static mut MySharable: Sharable = Sharable { field: 0 };

pub fn fn_mut(_: &mut Sharable) {}


pub fn main() {
    MySharable.foo();

    MySharable.foo_mut();
    //~^ ERROR mutable borrow of mutable static requires unsafe function or bloc

    MySharable.field = 2;
    //~^ ERROR: this use of mutable static requires unsafe function or block

    fn_mut(&mut MySharable);
    //~^ ERROR mutable borrow of mutable static requires unsafe function or block

    MyNonSharable.field = 2;
    //~^ ERROR: this use of mutable static requires unsafe function or block

    MySharable = Sharable {field: 1};
    //~^ ERROR: this use of mutable static requires unsafe function or block

    let _: &mut Sharable = &mut MySharable;
    //~^ ERROR mutable borrow of mutable static requires unsafe function or block

    let _ = &MyNonSharable.field;
    //~^ ERROR: this use of mutable static requires unsafe function or block

    let mut slc = ['a', 'c'];
    slc[MyNonSharable.field] = 'b';
    //~^ ERROR: this use of mutable static requires unsafe function or block

    slc[MySharable.field] = 'b';
}
