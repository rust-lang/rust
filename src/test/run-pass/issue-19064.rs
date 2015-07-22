// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct Foo(pub str);

impl Foo {
  fn print(&self) -> &str {
    match self {
      &Foo(ref s) => s,
    }
  }
}

fn main() {
    let f: &Foo = unsafe { std::mem::transmute("foo") };
    assert_eq!(f.print(), "foo");
}
