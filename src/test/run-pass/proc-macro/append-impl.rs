// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:append-impl.rs

#![allow(warnings)]

#[macro_use]
extern crate append_impl;

trait Append {
    fn foo(&self);
}

#[derive(PartialEq,
         Append,
         Eq)]
struct A {
    inner: u32,
}

fn main() {
    A { inner: 3 }.foo();
}
