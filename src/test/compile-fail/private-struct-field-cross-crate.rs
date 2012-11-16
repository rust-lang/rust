// aux-build:cci_class.rs
extern mod cci_class;
use cci_class::kitties::*;

fn main() {
  let nyan : cat = cat(52u, 99);
  assert (nyan.meows == 52u);   //~ ERROR field `meows` is private
}
