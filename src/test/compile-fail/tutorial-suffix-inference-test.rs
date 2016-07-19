// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x = 3;
    let y: i32 = 3;

    fn identity_u8(n: u8) -> u8 { n }
    fn identity_u16(n: u16) -> u16 { n }

    identity_u8(x);  // after this, `x` is assumed to have type `u8`
    identity_u16(x);
    //~^ ERROR mismatched types
    //~| expected u16, found u8
    identity_u16(y);
    //~^ ERROR mismatched types
    //~| expected u16, found i32

    let a = 3;

    fn identity_i(n: isize) -> isize { n }

    identity_i(a); // ok
    identity_u16(a);
    //~^ ERROR mismatched types
    //~| expected u16, found isize
}
