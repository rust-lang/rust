// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static A: [u8, ..1] = ['h' as u8];
static B: u8 = (&A)[0];
static C: &'static &'static &'static &'static [u8, ..1] = & & & &A;
static D: u8 = (&C)[0];

pub fn main() {
    assert!(B == A[0]);
    assert!(D == A[0]);
}
