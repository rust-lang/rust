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
//[mir]compile-flags: -Z borrowck=mir

fn test() {
    let w: &mut [isize];
    w[5] = 0; //[ast]~ ERROR use of possibly uninitialized variable: `*w` [E0381]
              //[mir]~^ ERROR [E0381]

    let mut w: &mut [isize];
    w[5] = 0; //[ast]~ ERROR use of possibly uninitialized variable: `*w` [E0381]
              //[mir]~^ ERROR [E0381]
}

fn main() { test(); }
