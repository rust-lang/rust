// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test an interesting corner case that ought to be legal (though the
// current code actually gets it wrong, see below): a fn that takes
// two arguments that are references with the same lifetime is in fact
// equivalent to a fn that takes two references with distinct
// lifetimes. This is true because the two functions can call one
// another -- effectively, the single lifetime `'a` is just inferred
// to be the intersection of the two distinct lifetimes.
//
// FIXME: However, we currently reject this example with an error,
// because of how we handle binders and equality in `relate_tys`.
//
// compile-flags:-Zno-leak-check

#![feature(nll)]

use std::cell::Cell;

fn make_cell_aa() -> Cell<for<'a> fn(&'a u32, &'a u32)> {
    panic!()
}

fn aa_eq_ab() {
    let a: Cell<for<'a, 'b> fn(&'a u32, &'b u32)> = make_cell_aa();
    //~^ ERROR higher-ranked subtype error
    drop(a);
}

fn main() { }
