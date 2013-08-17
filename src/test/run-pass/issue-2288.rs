// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait clam<A> {
  fn chowder(&self, y: A);
}
struct foo<A> {
  x: A,
}

impl<A> clam<A> for foo<A> {
  fn chowder(&self, _y: A) {
  }
}

fn foo<A>(b: A) -> foo<A> {
    foo {
        x: b
    }
}

fn f<A>(x: @clam<A>, a: A) {
  x.chowder(a);
}

pub fn main() {

  let c = foo(42);
  let d: @clam<int> = @c as @clam<int>;
  f(d, c.x);
}
