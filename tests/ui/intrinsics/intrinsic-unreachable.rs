// run-pass
#![feature(core_intrinsics)]

use std::intrinsics;

// See also src/test/run-make/intrinsic-unreachable.

unsafe fn f(x: usize) -> usize {
    match x {
        17 => 23,
        _ => intrinsics::unreachable(),
    }
}

fn main() {
    assert_eq!(unsafe { f(17) }, 23);
}
