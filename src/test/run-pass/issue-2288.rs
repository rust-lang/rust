// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait clam<A: Copy> {
  fn chowder(y: A);
}
struct foo<A: Copy> {
  x: A,
}

impl<A: Copy> foo<A> : clam<A> {
  fn chowder(y: A) {
  }
}

fn foo<A: Copy>(b: A) -> foo<A> {
    foo {
        x: b
    }
}

fn f<A: Copy>(x: clam<A>, a: A) {
  x.chowder(a);
}

fn main() {

  let c = foo(42);
  let d: clam<int> = c as clam::<int>;
  f(d, c.x);
}
