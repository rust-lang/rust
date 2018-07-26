// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn main() {
    let a0 = 0;
    let f = 1;
    let mut a1 = &a0;
    match (&a1,) {
        (&ref b0,) => {
            a1 = &f; //[ast]~ ERROR cannot assign
            //[mir]~^ ERROR cannot assign to `a1` because it is borrowed
            drop(b0);
        }
    }
}
