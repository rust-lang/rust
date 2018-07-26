// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Zborrowck=mir

fn test() {
    let v: isize;
    //[mir]~^ NOTE consider changing this to `mut v`
    loop {
        v = 1; //[ast]~ ERROR cannot assign twice to immutable variable
               //[mir]~^ ERROR cannot assign twice to immutable variable `v`
               //[ast]~| NOTE cannot assign twice to immutable variable
               //[mir]~| NOTE cannot assign twice to immutable variable
        v.clone(); // just to prevent liveness warnings
    }
}

fn main() {
}
