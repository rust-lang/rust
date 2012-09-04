// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_intrinsic.rs

// xfail-test

use cci_intrinsic;
import cci_intrinsic::atomic_xchg;

fn main() {
    let mut x = 1;
    atomic_xchg(&mut x, 5);
    assert x == 5;
}
