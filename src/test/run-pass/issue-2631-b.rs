// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:issue-2631-a.rs

extern mod req;
extern mod std;

use req::*;
use std::map::*;
use std::map::HashMap;

fn main() {
  let v = ~[@~"hi"];
  let m: req::header_map = HashMap();
  m.insert(~"METHOD", @dvec::from_vec(v));
  request::<int>(m);
}
