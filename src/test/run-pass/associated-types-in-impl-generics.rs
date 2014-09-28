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
    fn get(&self) -> &<Self as Get>::Value;
}

struct Struct {
    x: int,
}

impl Get for Struct {
    type Value = int;
    fn get(&self) -> &int {
        &self.x
    }
}

trait Grab {
    type U;
    fn grab(&self) -> &<Self as Grab>::U;
}

impl<T:Get> Grab for T {
    type U = <T as Get>::Value;
    fn grab(&self) -> &<T as Get>::Value {
        self.get()
    }
}

fn main() {
    let s = Struct {
        x: 100,
    };
    assert_eq!(*s.grab(), 100);
}

