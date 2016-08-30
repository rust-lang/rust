// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::usize;

struct dog {
    food: usize,
}

impl dog {
    pub fn chase_cat(&mut self) {
        let _f = || {
            //~^ first, the lifetime cannot outlive the lifetime  as defined on the block
            let p: &'static mut usize = &mut self.food;
            //~^ ERROR cannot infer an appropriate lifetime for borrow expression due to conflicting
            //~| ERROR cannot infer an appropriate lifetime for borrow expression due to conflicting
            //~| ERROR cannot infer an appropriate lifetime for borrow expression due to conflicting
            //~| NOTE cannot infer an appropriate lifetime
            //~| NOTE ...so that closure can access `self`
            //~| NOTE ...so that reference does not outlive borrowed content
            //~| NOTE but, the lifetime must be valid for the static lifetime...
            //~| NOTE but, the lifetime must be valid for the static lifetime...
            //~| NOTE but, the lifetime must be valid for the static lifetime...
            *p = 3;
        };
    }
}

fn main() {
}
