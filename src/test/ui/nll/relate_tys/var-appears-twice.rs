// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the NLL `relate_tys` code correctly deduces that a
// function returning always its first argument can be upcast to one
// that returns either first or second argument.

#![feature(nll)]
#![allow(warnings)]

use std::cell::Cell;

type DoubleCell<A> = Cell<(A, A)>;
type DoublePair<A> = (A, A);

fn make_cell<'b>(x: &'b u32) -> Cell<(&'static u32, &'b u32)> {
    panic!()
}

fn main() {
    let a: &'static u32 = &22;
    let b = 44;

    // Here we get an error because `DoubleCell<_>` requires the same type
    // on both parts of the `Cell`, and we can't have that.
    let x: DoubleCell<_> = make_cell(&b); //~ ERROR

    // Here we do not get an error because `DoublePair<_>` permits
    // variance on the lifetimes involved.
    let y: DoublePair<_> = make_cell(&b).get();
}
