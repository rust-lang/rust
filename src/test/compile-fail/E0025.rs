// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    a: u8,
    b: u8,
}

fn main() {
    let x = Foo { a:1, b:2 };
    let Foo { a: x, a: y, b: 0 } = x;
    //~^ ERROR field `a` bound multiple times in the pattern
    //~| NOTE multiple uses of `a` in pattern
    //~| NOTE first use of `a`
}
