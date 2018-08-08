// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Be smart about span of parenthesized expression in macro.

macro_rules! paren {
    ($e:expr) => (($e))
    //            ^^^^ do not highlight here
}

mod m {
    pub struct S {
        x: i32
    }
    pub fn make() -> S {
        S { x: 0 }
    }
}

fn main() {
    let s = m::make();
    paren!(s.x); //~ ERROR field `x` of struct `m::S` is private
    //     ^^^ highlight here
}
