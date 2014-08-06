// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

static mut NON_SHARABLE: NonSharable = NonSharable {
    field: 1,
    noshare: marker::NoShare,
};

static mut SHARABLE: Sharable = Sharable { field: 0 };

pub fn fn_mut(_: &mut Sharable) {}

pub fn main() {
    SHARABLE.foo();

    SHARABLE.foo_mut();
    //~^ ERROR: mutable use of static requires unsafe function or block

    SHARABLE.field = 2;
    //~^ ERROR: mutable use of static requires unsafe function or block

    fn_mut(&mut SHARABLE);
    //~^ ERROR mutable use of static requires unsafe function or block

    NON_SHARABLE.field = 2;
    //~^ ERROR: use of non-Share static mut requires unsafe function or block
    //~^^ ERROR: mutable use of static requires unsafe function or block

    SHARABLE = Sharable {field: 1};
    //~^ ERROR: mutable use of static requires unsafe function or block

    let _: &mut Sharable = &mut SHARABLE;
    //~^ ERROR mutable use of static requires unsafe function or block

    let _ = &NON_SHARABLE.field;
    //~^ ERROR: use of non-Share static mut requires unsafe function or block

    let mut slc = ['a', 'c'];
    slc[NON_SHARABLE.field] = 'b';
    //~^ ERROR: use of non-Share static mut requires unsafe function or block

    slc[SHARABLE.field] = 'b';
}
