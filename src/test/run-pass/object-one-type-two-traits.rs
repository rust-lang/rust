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

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::any::Any;

trait Wrap {
    fn get(&self) -> isize;
    fn wrap(self: Box<Self>) -> Box<Any+'static>;
}

impl Wrap for isize {
    fn get(&self) -> isize {
        *self
    }
    fn wrap(self: Box<isize>) -> Box<Any+'static> {
        self as Box<Any+'static>
    }
}

fn is<T:Any>(x: &Any) -> bool {
    x.is::<T>()
}

fn main() {
    let x = box 22isize as Box<Wrap>;
    println!("x={}", x.get());
    let y = x.wrap();
}
