// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct T(&'static [int]);

static A: T = T(&'static [5, 4, 3]);
static B: T = T(&[5, 4, 3]);
static C: T = T([5, 4, 3]);

pub fn main() {
    assert_eq!(A[0], 5);
    assert_eq!(B[1], 4);
    assert_eq!(C[2], 3);
}
