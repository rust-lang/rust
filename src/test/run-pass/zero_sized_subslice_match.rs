// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]

fn main() {
    let x = [(), ()];

    // The subslice used to go out of bounds for zero-sized array items, check that this doesn't
    // happen anymore
    match x {
        [_, ref y..] => assert_eq!(&x[1] as *const (), &y[0] as *const ())
    }
}
