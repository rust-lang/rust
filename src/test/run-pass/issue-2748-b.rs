// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn thing<'r>(x: &'r [int]) -> &'r [int] { x }

pub fn main() {
    let x = &[1,2,3];
    let y = x;
    let z = thing(x);
    assert_eq!(z[2], x[2]);
    assert_eq!(z[1], y[1]);
}
