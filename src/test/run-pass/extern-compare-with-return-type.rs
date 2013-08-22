// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that we can compare various kinds of extern fn signatures.

extern fn voidret1() {}
extern fn voidret2() {}

extern fn uintret() -> uint { 22 }

extern fn uintvoidret(x: uint) {}

extern fn uintuintuintuintret(x: uint, y: uint, z: uint) -> uint { x+y+z }

fn main() {
    assert_eq!(voidret1, voidret1);
    assert!(voidret1 != voidret2);

    assert_eq!(uintret, uintret);

    assert_eq!(uintvoidret, uintvoidret);

    assert_eq!(uintuintuintuintret, uintuintuintuintret);
}

