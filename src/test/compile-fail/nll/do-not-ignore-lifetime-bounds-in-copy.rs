// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the 'static bound from the Copy impl is respected. Regression test for #29149.

#![feature(nll)]

#[derive(Clone)] struct Foo<'a>(&'a u32);
impl Copy for Foo<'static> {}

fn main() {
    let s = 2;
    let a = Foo(&s); //~ ERROR `s` does not live long enough [E0597]
    drop(a);
    drop(a);
}
