// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

union U {
    a: u8
}

fn main() {
    let mut u = U { a: 10 }; // OK
    let a = u.a; //~ ERROR access to union field requires unsafe function or block
    u.a = 11; //~ ERROR access to union field requires unsafe function or block
    let U { a } = u; //~ ERROR matching on union field requires unsafe function or block
    if let U { a: 12 } = u {} //~ ERROR matching on union field requires unsafe function or block
    // let U { .. } = u; // OK
}
