// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// testing whether the lookup mechanism picks up types
// defined in the outside crate

// aux-build:issue-21221-4.rs

extern crate issue_21221_4;

struct Foo;

impl T for Foo {}
//~^ ERROR unresolved trait `T`
//~| HELP you can import it into scope: `use issue_21221_4::T;`

fn main() {
    println!("Hello, world!");
}
