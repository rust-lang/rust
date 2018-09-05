// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A;

impl A {
//~^ NOTE `Self` type implicitly declared here, on the `impl`
    fn banana(&mut self) {
        fn peach(this: &Self) {
        //~^ ERROR can't use type parameters from outer function
        //~| NOTE use of type variable from outer function
        //~| NOTE use a type here instead
        }
    }
}

fn main() {}
