// error-pattern:attempted access of field `nap` on type
// xfail-test Cross-crate impl method privacy doesn't work
// xfail-fast
// aux-build:cci_class_5.rs
extern mod cci_class_5;
use cci_class_5::kitties::*;

fn main() {
  let nyan : cat = cat(52, 99);
  nyan.nap();
}
