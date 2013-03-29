// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #2482.

pub fn main() {
    let v1: ~[int] = ~[10, 20, 30,];
    let v2: ~[int] = ~[10, 20, 30];
    assert!((v1[2] == v2[2]));
    let v3: ~[int] = ~[10,];
    let v4: ~[int] = ~[10];
    assert!((v3[0] == v4[0]));
}
