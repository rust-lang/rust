// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_intrinsic.rs

use cci_intrinsic;
use cci_intrinsic::atomic_xchg;

fn main() {
    let mut x = 1;
    atomic_xchg(&mut x, 5);
    assert x == 5;
}
