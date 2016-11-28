// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Can't use constants as tuple struct patterns

#![feature(associated_consts)]

const C1: i32 = 0;

struct S;

impl S {
    const C2: i32 = 0;
}

fn main() {
    if let C1(..) = 0 {} //~ ERROR expected tuple struct/variant, found constant `C1`
    if let S::C2(..) = 0 {}
    //~^ ERROR expected tuple struct/variant, found associated constant `<S>::C2`
}
