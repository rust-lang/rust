// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo {
    Bar(i32),
    Baz
}

struct S;

fn main() {
    match Foo::Baz {
        Foo::Bar => {}
        //~^ ERROR `Foo::Bar` does not name a unit variant, unit struct or a constant
        _ => {}
    }

    match S {
        S(()) => {}
        //~^ ERROR `S` does not name a tuple variant or a tuple struct
    }
}
