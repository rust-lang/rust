// aux-build:cci_class_5.rs
extern crate cci_class_5;
use cci_class_5::kitties::cat;

fn main() {
  let nyan : cat = cat(52, 99);
  nyan.nap();   //~ ERROR method `nap` is private
}
