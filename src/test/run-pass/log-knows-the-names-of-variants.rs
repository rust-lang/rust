// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum foo {
  a(uint),
  b(~str),
  c,
}

enum bar {
  d, e, f
}

pub fn main() {
    fail_unless_eq!(~"a(22u)", format!("{:?}", a(22u)));
    fail_unless_eq!(~"b(~\"hi\")", format!("{:?}", b(~"hi")));
    fail_unless_eq!(~"c", format!("{:?}", c));
    fail_unless_eq!(~"d", format!("{:?}", d));
}
