// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that binary operators allow subtyping on both the LHS and RHS,
// and as such do not introduce unnecesarily strict lifetime constraints.

use std::ops::Add;

struct Foo;

impl<'a> Add<&'a Foo> for &'a Foo {
    type Output = ();
    fn add(self, rhs: &'a Foo) {}
}

fn try_to_add(input: &Foo) {
    let local = Foo;

    // Manual reborrow worked even with invariant trait search.
    &*input + &local;

    // Direct use of the reference on the LHS requires additional
    // subtyping before searching (invariantly) for `LHS: Add<RHS>`.
    input + &local;
}

fn main() {
}
