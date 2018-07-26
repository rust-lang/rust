// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

#![allow(unused_variables)]
#![allow(unused_assignments)]

fn separate_arms() {
    // Here both arms perform assignments, but only is illegal.

    let mut x = None;
    match x {
        None => {
            // It is ok to reassign x here, because there is in
            // fact no outstanding loan of x!
            x = Some(0);
        }
        Some(ref r) => {
            x = Some(1); //[ast]~ ERROR cannot assign
            //[mir]~^ ERROR cannot assign to `x` because it is borrowed
            drop(r);
        }
    }
    x.clone(); // just to prevent liveness warnings
}

fn main() {}
