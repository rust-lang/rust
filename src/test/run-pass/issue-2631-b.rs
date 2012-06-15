// xfail-fast
// aux-build:issue-2631-a.rs

use req;
use std;

import req::*;
import std::map::*;
import std::map::str_hash;
import dvec;

fn main() {
  let v = [mut @"hi"];
  let m: req::header_map = str_hash();
  m.insert("METHOD", @dvec::from_vec(v));
  request::<int>(m);
}
