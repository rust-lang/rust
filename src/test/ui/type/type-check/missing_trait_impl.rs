// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
}

fn foo<T>(x: T, y: T) {
    let z = x + y; //~ ERROR binary operation `+` cannot be applied to type `T`
}

fn bar<T>(x: T) {
    x += x; //~ ERROR binary assignment operation `+=` cannot be applied to type `T`
}
