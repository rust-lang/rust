// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure casts between thin pointer <-> fat pointer are illegal.

pub trait Trait {}

fn main() {
    let a: &[i32] = &[1, 2, 3];
    let b: Box<[i32]> = Box::new([1, 2, 3]);

    a as usize; //~ ERROR non-scalar cast
    b as usize; //~ ERROR non-scalar cast

    let a: usize = 42;
    a as *const [i32]; //~ ERROR cast to fat pointer: `usize` as `*const [i32]`

    let a: *const u8 = &42;
    a as *const [u8]; //~ ERROR cast to fat pointer: `*const u8` as `*const [u8]`
}
