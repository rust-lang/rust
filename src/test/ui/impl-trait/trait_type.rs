// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct MyType;
struct MyType2;
struct MyType3;
struct MyType4;

impl std::fmt::Display for MyType {
   fn fmt(&self, x: &str) -> () { }
   //~^ ERROR method `fmt` has an incompatible type
}

impl std::fmt::Display for MyType2 {
   fn fmt(&self) -> () { }
   //~^ ERROR method `fmt` has 1 parameter
}

impl std::fmt::Display for MyType3 {
   fn fmt() -> () { }
   //~^ ERROR method `fmt` has a `&self` declaration in the trait
}

impl std::fmt::Display for MyType4 {}
//~^ ERROR not all trait items

fn main() {}
