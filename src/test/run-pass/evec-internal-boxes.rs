// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let x : [@int, ..5] = [@1,@2,@3,@4,@5];
    let _y : [@int, ..5] = [@1,@2,@3,@4,@5];
    let mut z = [@1,@2,@3,@4,@5];
    z = x;
    assert_eq!(*z[0], 1);
    assert_eq!(*z[4], 5);
}
