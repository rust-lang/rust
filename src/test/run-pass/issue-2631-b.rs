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
