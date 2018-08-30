// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum A {
    A1,
    A2,
    A3,
}

enum B {
    B1(String, String),
    B2(String, String),
}

fn main() {
    let a = A::A1;
    loop {
        let _ctor = match a {
            A::A3 => break,
            A::A1 => B::B1,
            A::A2 => B::B2,
        };
        break;
    }
}
