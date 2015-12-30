// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

#[rustc_mir]
pub fn foo(x: i8) -> i32 {
  match x {
    1...10 => 0,
    _ => 1,
  }
}

fn main() {
  assert_eq!(foo(0), 1);
  assert_eq!(foo(1), 0);
  assert_eq!(foo(2), 0);
  assert_eq!(foo(5), 0);
  assert_eq!(foo(9), 0);
  assert_eq!(foo(10), 0);
  assert_eq!(foo(11), 1);
  assert_eq!(foo(20), 1);
}
