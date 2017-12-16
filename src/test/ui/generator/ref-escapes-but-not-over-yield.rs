// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::cell::Cell;

fn foo(x: &i32) {
    // In this case, a reference to `b` escapes the generator, but not
    // because of a yield. We see that there is no yield in the scope of
    // `b` and give the more generic error message.
    let mut a = &3;
    let mut b = move || {
        yield();
        let b = 5;
        a = &b;
        //~^ ERROR `b` does not live long enough
    };
}

fn main() { }
