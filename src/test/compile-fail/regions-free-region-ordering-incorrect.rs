// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that free regions ordering only goes one way. That is,
// we have `&'a Node<'b, T>`, which implies that `'a <= 'b`,
// but not `'b <= 'a`. Hence returning `&self.val` (which has lifetime
// `'a`) where `'b` is expected yields an error.
//
// This test began its life as a test for issue #4325.

struct Node<'b, T> {
  val: T,
  next: Option<&'b Node<'b, T>>
}

impl<'b, T> Node<'b, T> {
  fn get<'a>(&'a self) -> &'b T {
    match self.next {
      Some(ref next) => next.get(),
      None => &self.val //~ ERROR cannot infer an appropriate lifetime
    }
  }
}

fn main() {}
