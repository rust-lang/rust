// xfail-fast
// aux-build:cci_class_4.rs
extern mod cci_class_4;
use cci_class_4::kitties::*;

fn main() {
  let nyan = cat(0u, 2, ~"nyan");
  nyan.eat();
  assert(!nyan.eat());
  for uint::range(1u, 10u) |_i| { nyan.speak(); };
  assert(nyan.eat());
}