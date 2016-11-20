// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait {}

// Make sure casts between thin-pointer <-> fat pointer obey RFC401
fn main() {
    let a: &[i32] = &[1, 2, 3];
    let b: Box<[i32]> = Box::new([1, 2, 3]);
    let p = a as *const [i32];
    let q = a.as_ptr();

    a as usize; //~ ERROR casting
    //~^ HELP cast through a raw pointer first
    a as isize; //~ ERROR casting
    a as i16; //~ ERROR casting `&[i32]` as `i16` is invalid
    a as u32; //~ ERROR casting `&[i32]` as `u32` is invalid
    b as usize; //~ ERROR non-scalar cast
    p as usize;
    //~^ ERROR casting
    //~^^ HELP cast through a thin pointer

    // #22955
    q as *const [i32]; //~ ERROR cannot cast

    // #21397
    let t: *mut (Trait + 'static) = 0 as *mut _; //~ ERROR casting
    let mut fail: *const str = 0 as *const str; //~ ERROR casting
}
