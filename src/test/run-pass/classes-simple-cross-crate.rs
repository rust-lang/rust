// xfail-fast
// aux-build:cci_class.rs
extern mod cci_class;
use cci_class::kitties::*;

fn main() {
  let nyan : cat = cat(52u, 99);
  let kitty = cat(1000u, 2);
  assert(nyan.how_hungry == 99);
  assert(kitty.how_hungry == 2);
}
