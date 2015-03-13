// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

fn main() {
    let f: Box<_>;
    let g;
    g = f;
    f = box g;
    //~^ ERROR the trait `core::ops::Place<Box<_>>` is not implemented for the type

    // (At one time, we produced a nicer error message like below.
    //  but right now, the desugaring produces the above error instead
    //  for the cyclic type here; its especially unfortunate because
    //  printed error leaves out the information necessary for one to
    //  deduce that the necessary type for the given impls *is*
    //  cyclic.)
    //
    // ^  ERROR mismatched types
    // | expected `_`
    // | found `Box<_>`
    // | cyclic type of infinite size
}
