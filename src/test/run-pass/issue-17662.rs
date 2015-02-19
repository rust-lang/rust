// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-17662.rs

extern crate "issue-17662" as i;

use std::marker;

struct Bar<'a> { m: marker::PhantomData<&'a ()> }

impl<'a> i::Foo<'a, uint> for Bar<'a> {
    fn foo(&self) -> uint { 5_usize }
}

pub fn main() {
    assert_eq!(i::foo(&Bar { m: marker::PhantomData }), 5);
}
