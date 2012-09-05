// xfail-fast: aux-build not compatible with fast
// aux-build:issue_2723_a.rs

use issue_2723_a;
use issue_2723_a::*;

fn main() unsafe {
  f(~[2]);
}