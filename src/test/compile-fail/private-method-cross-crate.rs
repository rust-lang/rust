// error-pattern:attempted access of field nap on type
// xfail-fast
// aux-build:cci_class_5.rs
use cci_class_5;
import cci_class_5::kitties::*;

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.nap();
}
