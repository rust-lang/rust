// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that a constant-evaluation underflow highlights the correct
// spot (where the underflow occurred).

const ONE: usize = 1;
const TWO: usize = 2;

fn main() {
    let a: [i8; ONE - TWO] = unimplemented!();
    //~^ ERROR constant evaluation error [E0080]
    //~| attempt to subtract with overflow
}
