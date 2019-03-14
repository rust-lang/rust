// run-pass
// aux-build:issue-2316-a.rs
// aux-build:issue-2316-b.rs

// pretty-expanded FIXME #23616

extern crate issue_2316_b;
use issue_2316_b::cloth;

pub fn main() {
  let _c: cloth::fabric = cloth::fabric::calico;
}
