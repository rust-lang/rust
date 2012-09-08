// error-pattern:attempted access of field `nap` on type
// xfail-fast
// xfail-test
// aux-build:cci_class_5.rs
use cci_class_5;
use cci_class_5::kitties::*;

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.nap();
}
