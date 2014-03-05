// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Eq, Show)]
struct Foo(int, int, ~str);

pub fn main() {
  let a1 = Foo(5, 6, ~"abc");
  let a2 = Foo(5, 6, ~"abc");
  let b = Foo(5, 7, ~"def");

  assert!(a1 == a1);
  assert!(a2 == a1);
  assert!(!(a1 == b));

  assert!(a1 != b);
  assert!(!(a1 != a1));
  assert!(!(a2 != a1));
}
