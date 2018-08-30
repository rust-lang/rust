// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(unused_variables)]

trait Bar {
  fn noop(&self);
}
impl Bar for u8 {
  fn noop(&self) {}
}

fn main() {
    let (a, b) = (&5u8 as &Bar, &9u8 as &Bar);
    let (c, d): (&Bar, &Bar) = (a, b);

    let (a, b) = (Box::new(5u8) as Box<Bar>, Box::new(9u8) as Box<Bar>);
    let (c, d): (&Bar, &Bar) = (&*a, &*b);

    let (c, d): (&Bar, &Bar) = (&5, &9);
}
