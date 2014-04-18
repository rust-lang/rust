// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

// aux-build:issue-2631-a.rs

extern crate collections;
extern crate req;

use req::request;
use std::cell::RefCell;
use collections::HashMap;

pub fn main() {
  let v = vec!(@"hi".to_owned());
  let mut m: req::header_map = HashMap::new();
  m.insert("METHOD".to_owned(), @RefCell::new(v));
  request::<int>(&m);
}
