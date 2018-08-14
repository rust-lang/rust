// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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
    let foo = &mut 1;

    let &mut x = foo;
    x += 1; //[ast]~ ERROR cannot assign twice to immutable variable
            //[mir]~^ ERROR cannot assign twice to immutable variable `x`

    // explicitly mut-ify internals
    let &mut mut x = foo;
    x += 1;

    // check borrowing is detected successfully
    let &mut ref x = foo;
    *foo += 1; //[ast]~ ERROR cannot assign to `*foo` because it is borrowed
    //[mir]~^ ERROR cannot assign to `*foo` because it is borrowed
    drop(x);
}
