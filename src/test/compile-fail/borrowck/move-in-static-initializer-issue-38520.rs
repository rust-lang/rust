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

// Regression test for #38520. Check that moves of `Foo` are not
// permitted as `Foo` is not copy (even in a static/const
// initializer).

#![feature(const_fn)]

struct Foo(usize);

const fn get(x: Foo) -> usize {
    x.0
}

const X: Foo = Foo(22);
static Y: usize = get(*&X); //[ast]~ ERROR E0507
                            //[mir]~^ ERROR [E0507]
const Z: usize = get(*&X); //[ast]~ ERROR E0507
                           //[mir]~^ ERROR [E0507]

fn main() {
}
