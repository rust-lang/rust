// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_types)]

trait Get {
    type Value;
    fn get(&self) -> <Self as Get>::Value;
}

fn get(x: int) -> <int as Get>::Value {}
//~^ ERROR this associated type is not allowed in this context

struct Struct {
    x: int,
}

impl Struct {
    fn uhoh<T>(foo: <T as Get>::Value) {}
    //~^ ERROR this associated type is not allowed in this context
}

fn main() {
}

