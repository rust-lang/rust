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
// spot (where the underflow occurred), while also providing the
// overall context for what caused the evaluation.

const ONE: usize = 1;
const TWO: usize = 2;
const LEN: usize = ONE - TWO;
//~^ ERROR array length constant evaluation error: attempted to sub with overflow [E0250]

fn main() {
    let a: [i8; LEN] = unimplemented!();
    //~^ NOTE for array length here
}
