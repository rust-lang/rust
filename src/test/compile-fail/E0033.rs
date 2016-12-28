// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait SomeTrait {
    fn foo();
}

fn main() {
    let trait_obj: &SomeTrait = SomeTrait;
    //~^ ERROR expected value, found trait `SomeTrait`
    //~| NOTE not a value
    //~| ERROR E0038
    //~| method `foo` has no receiver
    //~| NOTE the trait `SomeTrait` cannot be made into an object

    let &invalid = trait_obj;
    //~^ ERROR E0033
    //~| NOTE type `&SomeTrait` cannot be dereferenced
}
