// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that negating unsigned integers is gated by `negate_unsigned` feature
// gate

struct S;
impl std::ops::Neg for S {
    type Output = u32;
    fn neg(self) -> u32 { 0 }
}

const _MAX: usize = -1;
//~^ ERROR unary negation of unsigned integers may be removed in the future

fn main() {
    let a = -1;
    //~^ ERROR unary negation of unsigned integers may be removed in the future
    let _b : u8 = a; // for infering variable a to u8.

    -a;
    //~^ ERROR unary negation of unsigned integers may be removed in the future

    let _d = -1u8;
    //~^ ERROR unary negation of unsigned integers may be removed in the future

    for _ in -10..10u8 {}
    //~^ ERROR unary negation of unsigned integers may be removed in the future

    -S; // should not trigger the gate; issue 26840
}
