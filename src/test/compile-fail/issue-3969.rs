// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Bike {
    name: ~str,
}

trait BikeMethods {
    fn woops(&const self) -> ~str;
}

pub impl Bike : BikeMethods {
    static fn woops(&const self) -> ~str { ~"foo" }
    //~^ ERROR method `woops` is declared as static in its impl, but not in its trait
}

pub fn main() {
}
