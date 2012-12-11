// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test rules governing higher-order pure fns.

fn take<T>(_v: T) {}

fn assign_to_pure(x: pure fn(), y: fn(), z: unsafe fn()) {
    take::<pure fn()>(x);
    take::<pure fn()>(y); //~ ERROR expected pure fn but found impure fn
    take::<pure fn()>(z); //~ ERROR expected pure fn but found unsafe fn
}

fn assign_to_impure(x: pure fn(), y: fn(), z: unsafe fn()) {
    take::<fn()>(x);
    take::<fn()>(y);
    take::<fn()>(z); //~ ERROR expected impure fn but found unsafe fn
}

fn assign_to_unsafe(x: pure fn(), y: fn(), z: unsafe fn()) {
    take::<unsafe fn()>(x);
    take::<unsafe fn()>(y);
    take::<unsafe fn()>(z);
}

fn assign_to_pure2(x: pure fn@(), y: fn@(), z: unsafe fn@()) {
    take::<pure fn()>(x);
    take::<pure fn()>(y); //~ ERROR expected pure fn but found impure fn
    take::<pure fn()>(z); //~ ERROR expected pure fn but found unsafe fn

    take::<pure fn~()>(x); //~ ERROR expected ~ closure, found @ closure
    take::<pure fn~()>(y); //~ ERROR expected ~ closure, found @ closure
    take::<pure fn~()>(z); //~ ERROR expected ~ closure, found @ closure

    take::<unsafe fn~()>(x); //~ ERROR expected ~ closure, found @ closure
    take::<unsafe fn~()>(y); //~ ERROR expected ~ closure, found @ closure
    take::<unsafe fn~()>(z); //~ ERROR expected ~ closure, found @ closure
}

fn main() {
}
