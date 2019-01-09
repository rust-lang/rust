// run-pass
// aux-build:cci_intrinsic.rs


extern crate cci_intrinsic;
use cci_intrinsic::atomic_xchg;

pub fn main() {
    let mut x = 1;
    atomic_xchg(&mut x, 5);
    assert_eq!(x, 5);
}
