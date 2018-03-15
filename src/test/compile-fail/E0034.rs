// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Test;

trait Trait1 {
    fn foo();
}

trait Trait2 {
    fn foo();
}

impl Trait1 for Test {
    fn foo() {}
    //~^ NOTE candidate #1 is defined in an impl of the trait `Trait1` for the type `Test`
}

impl Trait2 for Test {
    fn foo() {}
    //~^ NOTE candidate #2 is defined in an impl of the trait `Trait2` for the type `Test`
}

fn main() {
    Test::foo() //~ ERROR multiple applicable items in scope
    //~| NOTE multiple `foo` found
}
