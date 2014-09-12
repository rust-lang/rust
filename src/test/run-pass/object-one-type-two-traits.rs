// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing creating two vtables with the same self type, but different
// traits.

use std::any::Any;
use std::any::AnyRefExt;

trait Wrap {
    fn get(&self) -> int;
    fn wrap(self: Box<Self>) -> Box<Any+'static>;
}

impl Wrap for int {
    fn get(&self) -> int {
        *self
    }
    fn wrap(self: Box<int>) -> Box<Any+'static> {
        self as Box<Any+'static>
    }
}

fn is<T:'static>(x: &Any) -> bool {
    x.is::<T>()
}

fn main() {
    let x = box 22i as Box<Wrap>;
    println!("x={}", x.get());
    let y = x.wrap();
}
