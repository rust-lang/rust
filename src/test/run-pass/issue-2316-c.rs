// xfail-fast - check-fast doesn't understand aux-build
// aux-build:issue_2316_a.rs
// aux-build:issue_2316_b.rs

extern mod issue_2316_b;
use issue_2316_b::cloth;

fn main() {
  let _c: cloth::fabric = cloth::calico;
}