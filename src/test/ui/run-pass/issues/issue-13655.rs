// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits, unboxed_closures)]
use std::ops::Fn;

struct Foo<T>(T);

impl<T: Copy> Fn<()> for Foo<T> {
    extern "rust-call" fn call(&self, _: ()) -> T {
      match *self {
        Foo(t) => t
      }
    }
}

impl<T: Copy> FnMut<()> for Foo<T> {
    extern "rust-call" fn call_mut(&mut self, _: ()) -> T {
        self.call(())
    }
}

impl<T: Copy> FnOnce<()> for Foo<T> {
    type Output = T;

    extern "rust-call" fn call_once(self, _: ()) -> T {
        self.call(())
    }
}

fn main() {
  let t: u8 = 1;
  println!("{}", Foo(t)());
}
