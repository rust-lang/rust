// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:foo_trait_default_impl.rs

#![feature(specialization)]

extern crate foo_trait_default_impl;

use foo_trait_default_impl::*;

struct MyStruct;

fn main() {
    MyStruct.foo_two(); //~ NOTE the function call is here
}
