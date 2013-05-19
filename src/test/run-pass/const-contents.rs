// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #570

static lsl : int = 1 << 2;
static add : int = 1 + 2;
static addf : float = 1.0f + 2.0f;
static not : int = !0;
static notb : bool = !true;
static neg : int = -(1);

pub fn main() {
    assert_eq!(lsl, 4);
    assert_eq!(add, 3);
    assert_eq!(addf, 3.0f);
    assert_eq!(not, -1);
    assert_eq!(notb, false);
    assert_eq!(neg, -1);
}
