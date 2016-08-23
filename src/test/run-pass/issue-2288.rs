// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

trait clam<A> {
  fn chowder(&self, y: A);
}

#[derive(Copy, Clone)]
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

fn f<A>(x: Box<clam<A>>, a: A) {
  x.chowder(a);
}

pub fn main() {

  let c = foo(42);
  let d: Box<clam<isize>> = box c as Box<clam<isize>>;
  f(d, c.x);
}
