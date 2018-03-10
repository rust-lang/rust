// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const MAX_ITEM: usize = 10;

fn foo_bar() {}

fn foo(c: esize) {} // Misspelled primitive type name.
//~^ ERROR cannot find

enum Bar { }

type A = Baz; // Misspelled type name.
//~^ ERROR cannot find
type B = Opiton<u8>; // Misspelled type name from the prelude.
//~^ ERROR cannot find

mod m {
    type A = Baz; // No suggestion here, Bar is not visible
    //~^ ERROR cannot find

    pub struct First;
    pub struct Second;
}

fn main() {
    let v = [0u32; MAXITEM]; // Misspelled constant name.
    //~^ ERROR cannot find
    foobar(); // Misspelled function name.
    //~^ ERROR cannot find
    let b: m::first = m::second; // Misspelled item in module.
    //~^ ERROR cannot find
    //~| ERROR cannot find
}
