// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that a closure which mutates a local variable
// cannot also be supplied a borrowed version of that
// variable's contents. Issue #11192.


struct Foo {
  x: int
}

impl Drop for Foo {
  fn drop(&mut self) {
    println!("drop {}", self.x);
  }
}

fn main() {
  let mut ptr = box Foo { x: 0 };
  let test = |foo: &Foo| {
    ptr = box Foo { x: ptr.x + 1 };
  };
  test(&*ptr); //~ ERROR cannot borrow `*ptr`
}
