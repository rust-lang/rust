// xfail-fast - check-fast doesn't understand aux-build
// aux-build:issue_2316_a.rs
// aux-build:issue_2316_b.rs

use issue_2316_b;
import issue_2316_b::cloth;

fn main() {
  let _c: cloth::fabric = cloth::calico;
}