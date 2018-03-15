// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-36708.rs

extern crate issue_36708 as lib;

struct Bar;

impl lib::Foo for Bar {
    fn foo<T>() {}
    //~^ ERROR E0049
    //~| NOTE found 1 type parameter, expected 0
}

fn main() {}
