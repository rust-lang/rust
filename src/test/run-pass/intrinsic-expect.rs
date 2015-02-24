// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::intrinsics::expect8;

fn check8(x: u8) -> u8 {
    let b = (x > 0) as u8;
    if unsafe { expect8(b, 1) == 1 } {
        10
    } else {
        5
    }
}

fn main() {
    let x = check8(1);
    assert_eq!(x, 10);
    let x = check8(2);
    assert_eq!(x, 10);
    let x = check8(5);
    assert_eq!(x, 10);
    let x = check8(10);
    assert_eq!(x, 10);
    let x = check8(-1);
    assert_eq!(x, 5);

}
