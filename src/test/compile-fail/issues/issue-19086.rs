// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use Foo::FooB;

enum Foo {
    FooB { x: i32, y: i32 }
}

fn main() {
    let f = FooB { x: 3, y: 4 };
    match f {
        FooB(a, b) => println!("{} {}", a, b),
        //~^ ERROR expected tuple struct/variant, found struct variant `FooB`
    }
}
