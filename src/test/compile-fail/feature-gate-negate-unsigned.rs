// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that negating unsigned integers doesn't compile

struct S;
impl std::ops::Neg for S {
    type Output = u32;
    fn neg(self) -> u32 { 0 }
}

fn main() {
    let _max: usize = -1;
    //~^ ERROR cannot apply unary operator `-` to type `usize`

    let x = 5u8;
    let _y = -x;
    //~^ ERROR cannot apply unary operator `-` to type `u8`

    -S; // should not trigger the gate; issue 26840
}
