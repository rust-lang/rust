// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the visibility modifier is included in the span of foreign items.

extern {
  fn foo();

  pub //~ ERROR the name `foo` is defined multiple times
  fn foo();

  pub //~ ERROR the name `foo` is defined multiple times
  static mut foo: u32;
}

fn main() {
}
