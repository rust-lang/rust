// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(asm)]

#[cfg(target_arch = "x86")]
#[cfg(target_arch = "x86_64")]
pub fn main() {
    let mut x: int = 0;
    let y: int;
    let z: int;
    unsafe {
        // Output expressions are evaluated before input expressions.
        asm!("mov $1, $0" : "=r"(*{y = 3; &mut x}) : "r"({z = y; 7}));
    }
    assert_eq!(x, 7);
    assert_eq!(z, 3);
}

#[cfg(not(target_arch = "x86"), not(target_arch = "x86_64"))]
pub fn main() {}
