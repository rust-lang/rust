// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #47153: constants in a generic context (such as
// a trait) used to ICE.

#![feature(nll)]
#![allow(warnings)]

trait Foo {
    const B: bool = true;
}

struct Bar<T> { x: T }

impl<T> Bar<T> {
    const B: bool = true;
}

fn main() { }
