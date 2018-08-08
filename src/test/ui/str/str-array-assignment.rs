// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
  let s = "abc";
  let t = if true { s[..2] } else { s };
  //~^ ERROR if and else have incompatible types
  let u: &str = if true { s[..2] } else { s };
  //~^ ERROR mismatched types
  let v = s[..2];
  //~^ ERROR the size for values of type
  let w: &str = s[..2];
  //~^ ERROR mismatched types
}
