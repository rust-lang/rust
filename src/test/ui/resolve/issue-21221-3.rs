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

// aux-build:issue-21221-3.rs

extern crate issue_21221_3;

struct Foo;

// NOTE: This shows only traits accessible from the current
// crate, thus the two private entities:
//   `issue_21221_3::outer::private_module::OuterTrait` and
//   `issue_21221_3::outer::public_module::OuterTrait`
// are hidden from the view.
impl OuterTrait for Foo {}
//~^ ERROR unresolved trait `OuterTrait`
//~| HELP you can import it into scope: `use issue_21221_3::outer::OuterTrait;`
fn main() {
    println!("Hello, world!");
}
