// aux-build:cci_class.rs
extern crate cci_class;
use cci_class::kitties::cat;

fn main() {
  let nyan : cat = cat(52, 99);
  assert_eq!(nyan.meows, 52);
  //~^ ERROR field `meows` of struct `cci_class::kitties::cat` is private
}
