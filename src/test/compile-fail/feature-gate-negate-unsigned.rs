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

// FIXME(eddyb) move this back to a `-1` literal when
// MIR building stops eagerly erroring in that case.
const _MAX: usize = -(2 - 1);
//~^ WARN unary negation of unsigned integer
//~| ERROR unary negation of unsigned integer
//~| HELP use a cast or the `!` operator

fn main() {
    let x = 5u8;
    let _y = -x; //~ ERROR unary negation of unsigned integer
    //~^ HELP use a cast or the `!` operator
    -S; // should not trigger the gate; issue 26840
}
