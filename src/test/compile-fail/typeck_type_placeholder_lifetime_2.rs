// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks that the `_` type placeholder does not react
// badly if put as a lifetime parameter.

struct Foo<'a, T> {
    r: &'a T
}

pub fn main() {
    let c: Foo<_, uint> = Foo { r: &5 };
    //~^ ERROR wrong number of type arguments: expected 1 but found 2
}
