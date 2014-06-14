// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

use std::gc::Gc;

pub struct Foo {
    a: int,
}

struct Bar<'a> {
    a: Box<Option<int>>,
    b: &'a Foo,
}

fn check(a: Gc<Foo>) {
    let _ic = Bar{ b: a, a: box None };
}

pub fn main(){}
