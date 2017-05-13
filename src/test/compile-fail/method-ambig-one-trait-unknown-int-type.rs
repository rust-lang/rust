// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we invoking `foo()` successfully resolves to the trait `foo`
// (prompting the mismatched types error) but does not influence the choice
// of what kind of `Vec` we have, eventually leading to a type error.

trait foo {
    fn foo(&self) -> isize;
}

impl foo for Vec<usize> {
    fn foo(&self) -> isize {1}
}

impl foo for Vec<isize> {
    fn foo(&self) -> isize {2}
}

// This is very hokey: we have heuristics to suppress messages about
// type annotations required. But placing these two bits of code into
// distinct functions, in this order, causes us to print out both
// errors I'd like to see.

fn m1() {
    // we couldn't infer the type of the vector just based on calling foo()...
    let mut x = Vec::new();
    //~^ ERROR unable to infer enough type information about `T` [E0282]
    x.foo();
}

fn m2() {
    let mut x = Vec::new();

    // ...but we still resolved `foo()` to the trait and hence know the return type.
    let y: usize = x.foo(); //~ ERROR mismatched types
}

fn main() { }
