// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Some traits can be derived for unions.

#![feature(untagged_unions)]

#[derive(
    Copy,
)]
union U {
    a: u8,
    b: u16,
}

impl Clone for U {
    fn clone(&self) -> Self { *self }
}

fn main() {
    let u = U { b: 0 };
    let u1 = u;
    let u2 = u.clone();
}
