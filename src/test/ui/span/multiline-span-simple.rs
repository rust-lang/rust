// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(a: u32, b: u32) {
    a + b;
}

fn bar(a: u32, b: u32) {
    a + b;
}

fn main() {
    let x = 1;
    let y = 2;
    let z = 3;
    foo(1 as u32 +

        bar(x,

            y),

        z)
}
