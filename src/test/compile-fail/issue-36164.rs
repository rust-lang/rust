// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    bar: u8
}

struct Bar {
}

fn main() {
    let f = Foo { bar: 22 };
    f.xxx;
    //~^ ERROR no field `xxx` on type `Foo`
    //~| NOTE did you mean `bar`?
    let g = Bar { };
    g.yyy;
    //~^ ERROR no field `yyy` on type `Bar`
    //~| NOTE unknown field
}
