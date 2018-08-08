// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that binary operators consume their arguments

use std::ops::{Add, Sub, Mul, Div, Rem, BitAnd, BitXor, BitOr, Shl, Shr};

fn add<A: Add<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs + rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn sub<A: Sub<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs - rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn mul<A: Mul<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs * rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn div<A: Div<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs / rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn rem<A: Rem<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs % rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn bitand<A: BitAnd<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs & rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn bitor<A: BitOr<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs | rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn bitxor<A: BitXor<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs ^ rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn shl<A: Shl<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs << rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn shr<A: Shr<B, Output=()>, B>(lhs: A, rhs: B) {
    lhs >> rhs;
    drop(lhs);  //~ ERROR use of moved value: `lhs`
    drop(rhs);  //~ ERROR use of moved value: `rhs`
}

fn main() {}
