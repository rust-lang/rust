// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test case from #39963.

#![feature(nll)]

#[derive(Clone)]
struct Foo(Option<Box<Foo>>, Option<Box<Foo>>);

fn test(f: &mut Foo) {
  match *f {
    Foo(Some(ref mut left), Some(ref mut right)) => match **left {
      Foo(Some(ref mut left), Some(ref mut right)) => panic!(),
      _ => panic!(),
    },
    _ => panic!(),
  }
}

fn main() {
}
