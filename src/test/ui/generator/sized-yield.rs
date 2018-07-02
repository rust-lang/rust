// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait)]

use std::ops::Generator;

fn main() {
   let s = String::from("foo");
   let mut gen = move || {
   //~^ ERROR the size for value values of type
       yield s[..];
   };
   unsafe { gen.resume(); }
   //~^ ERROR the size for value values of type
}
