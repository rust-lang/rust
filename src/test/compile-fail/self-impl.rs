// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that unsupported uses of `Self` in impls don't crash

struct Bar;

trait Foo {
    type Baz;
}

trait SuperFoo {
    type SuperBaz;
}

impl Foo for Bar {
    type Baz = bool;
}

impl SuperFoo for Bar {
    type SuperBaz = bool;
}

impl Bar {
    fn f() {
        let _: <Self>::Baz = true;
        //~^ ERROR ambiguous associated type
        //~| NOTE ambiguous associated type
        //~| NOTE specify the type using the syntax `<Bar as Trait>::Baz`
        let _: Self::Baz = true;
        //~^ ERROR ambiguous associated type
        //~| NOTE ambiguous associated type
        //~| NOTE specify the type using the syntax `<Bar as Trait>::Baz`
    }
}

fn main() {}
