// run-pass
// aux-build:cci_class_2.rs

extern crate cci_class_2;
use cci_class_2::kitties::cat;

pub fn main() {
  let nyan : cat = cat(52, 99);
  let kitty = cat(1000, 2);
  assert_eq!(nyan.how_hungry, 99);
  assert_eq!(kitty.how_hungry, 2);
  nyan.speak();
}
