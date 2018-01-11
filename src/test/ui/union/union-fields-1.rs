// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(dead_code)]

union U1 {
    a: u8, // should not be reported
    b: u8, // should not be reported
    c: u8, //~ ERROR field is never used
}
union U2 {
    a: u8, //~ ERROR field is never used
    b: u8, // should not be reported
    c: u8, // should not be reported
}
union NoDropLike { a: u8 } //~ ERROR field is never used

union U {
    a: u8, // should not be reported
    b: u8, // should not be reported
    c: u8, //~ ERROR field is never used
}
type A = U;

fn main() {
    let u = U1 { a: 0 };
    let _a = unsafe { u.b };

    let u = U2 { c: 0 };
    let _b = unsafe { u.b };

    let _u = NoDropLike { a: 10 };
    let u = A { a: 0 };
    let _b = unsafe { u.b };
}
