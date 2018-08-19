// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions:rpass1 rpass2

#[cfg(rpass1)]
pub static A: u8 = 42;
#[cfg(rpass2)]
pub static A: u8 = 43;

static B: &u8 = &C.1;
static C: (&&u8, u8) = (&B, A);

fn main() {
    assert_eq!(*B, A);
    assert_eq!(**C.0, A);
    assert_eq!(C.1, A);
}
