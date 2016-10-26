// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait T1 {}
trait T2 {}

trait Foo {
    type A: T1; //~ NOTE: ambiguous `A` from `Foo`
}

trait Bar : Foo {
    type A: T2; //~ NOTE: ambiguous `A` from `Bar`
    fn do_something() {
        let _: Self::A;
        //~^ ERROR E0221
        //~| NOTE ambiguous associated type `A`
    }
}

trait T3 {}

trait My : std::str::FromStr {
    type Err: T3; //~ NOTE: ambiguous `Err` from `My`
    fn test() {
        let _: Self::Err;
        //~^ ERROR E0221
        //~| NOTE ambiguous associated type `Err`
        //~| NOTE associated type `Self` could derive from `std::str::FromStr`
    }
}

fn main() {
}
