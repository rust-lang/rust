// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir -Z two-phase-borrows

// This is the "goto example" for why we want two phase borrows.

fn main() {
    let mut v = vec![0, 1, 2];
    v.push(v.len());
    assert_eq!(v, [0, 1, 2, 3]);
}
