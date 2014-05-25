// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cci_class_cast.rs
extern crate cci_class_cast;

use std::to_str::ToStr;
use cci_class_cast::kitty::cat;

fn print_out(thing: Box<ToStr>, expected: String) {
  let actual = thing.to_str();
  println!("{}", actual);
  assert_eq!(actual.to_strbuf(), expected);
}

pub fn main() {
  let nyan: Box<ToStr> = box cat(0u, 2, "nyan".to_strbuf()) as Box<ToStr>;
  print_out(nyan, "nyan".to_strbuf());
}
