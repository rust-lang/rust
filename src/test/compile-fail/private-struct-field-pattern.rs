// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use a::Foo;

mod a {
    pub struct Foo {
        x: int
    }

    pub fn make() -> Foo {
        Foo { x: 3 }
    }
}

fn main() {
    match a::make() {
        Foo { x: _ } => {}  //~ ERROR field `x` of struct `a::Foo` is private
    }
}
