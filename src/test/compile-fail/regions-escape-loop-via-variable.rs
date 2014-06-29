// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x = 3i;

    // Here, the variable `p` gets inferred to a type with a lifetime
    // of the loop body.  The regionck then determines that this type
    // is invalid.
    let mut p = &x;

    loop {
        let x = 1i + *p;
        p = &x; //~ ERROR `x` does not live long enough
    }
}
