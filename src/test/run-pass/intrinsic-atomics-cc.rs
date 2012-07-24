// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_intrinsic.rs

// xfail-check

use cci_intrinsic;
import cci_intrinsic::atomic_xchng;

fn main() {
    let mut x = 1;
    atomic_xchng(x, 5);
    assert x == 5;
}
