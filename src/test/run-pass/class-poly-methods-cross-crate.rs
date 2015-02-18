// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cci_class_6.rs
extern crate cci_class_6;
use cci_class_6::kitties::cat;

pub fn main() {
  let mut nyan : cat<char> = cat::<char>(52_usize, 99, vec!('p'));
  let mut kitty = cat(1000_usize, 2, vec!("tabby".to_string()));
  assert_eq!(nyan.how_hungry, 99);
  assert_eq!(kitty.how_hungry, 2);
  nyan.speak(vec!(1_usize,2_usize,3_usize));
  assert_eq!(nyan.meow_count(), 55_usize);
  kitty.speak(vec!("meow".to_string(), "mew".to_string(), "purr".to_string(), "chirp".to_string()));
  assert_eq!(kitty.meow_count(), 1004_usize);
}
