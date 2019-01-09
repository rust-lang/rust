// run-pass
// aux-build:issue_2316_a.rs
// aux-build:issue_2316_b.rs

// pretty-expanded FIXME #23616

extern crate issue_2316_b;
use issue_2316_b::cloth;

pub fn main() {
  let _c: cloth::fabric = cloth::fabric::calico;
}
