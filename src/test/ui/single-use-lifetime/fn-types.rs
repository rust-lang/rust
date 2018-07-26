// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(single_use_lifetimes)]
#![allow(dead_code)]
#![allow(unused_variables)]

// Test that we DO warn when lifetime name is used only
// once in a fn argument.

struct Foo {
  a: for<'a> fn(&'a u32), //~ ERROR `'a` only used once
  b: for<'a> fn(&'a u32, &'a u32), // OK, used twice.
  c: for<'a> fn(&'a u32) -> &'a u32, // OK, used twice.
  d: for<'a> fn() -> &'a u32, // OK, used only in return type.
    //~^ ERROR return type references lifetime `'a`, which is not constrained by the fn input types
}

fn main() { }
