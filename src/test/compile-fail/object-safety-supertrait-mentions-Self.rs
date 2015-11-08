// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we correctly prevent users from making trait objects
// form traits that make use of `Self` in an argument or return position.

trait Bar<T> {
    fn bar(&self, x: &T);
}

trait Baz : Bar<Self> {
}

fn make_bar<T:Bar<u32>>(t: &T) -> &Bar<u32> {
    t
}

fn make_baz<T:Baz>(t: &T) -> &Baz {
    //~^ ERROR E0038
    //~| NOTE the trait cannot use `Self` as a type parameter in the supertrait listing
    t
}

fn main() {
}
