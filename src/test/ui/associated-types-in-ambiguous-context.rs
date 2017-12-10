// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Get {
    type Value;
    fn get(&self) -> <Self as Get>::Value;
}

fn get<T:Get,U:Get>(x: T, y: U) -> Get::Value {}
//~^ ERROR ambiguous associated type
//~| NOTE ambiguous associated type
//~| NOTE specify the type using the syntax `<Type as Get>::Value`

trait Grab {
    type Value;
    fn grab(&self) -> Grab::Value;
    //~^ ERROR ambiguous associated type
    //~| NOTE ambiguous associated type
    //~| NOTE specify the type using the syntax `<Type as Grab>::Value`
}

type X = std::ops::Deref::Target;
//~^ ERROR ambiguous associated type
//~| NOTE ambiguous associated type
//~| NOTE specify the type using the syntax `<Type as std::ops::Deref>::Target`

fn main() {
}
