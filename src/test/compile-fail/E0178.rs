// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {}

struct Bar<'a> {
    w: &'a Foo + Copy,
    //~^ ERROR E0178
    //~| NOTE expected a path
    x: &'a Foo + 'a,
    //~^ ERROR E0178
    //~| NOTE expected a path
    //~| ERROR at least one non-builtin trait is required for an object type
    y: &'a mut Foo + 'a,
    //~^ ERROR E0178
    //~| NOTE expected a path
    //~| ERROR at least one non-builtin trait is required for an object type
    z: fn() -> Foo + 'a,
    //~^ ERROR E0178
    //~| NOTE expected a path
    //~| ERROR at least one non-builtin trait is required for an object type
}

fn main() {
}
