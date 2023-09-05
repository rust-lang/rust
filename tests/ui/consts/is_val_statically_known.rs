// run-pass

#![feature(core_intrinsics)]
#![feature(is_val_statically_known)]

use std::intrinsics::is_val_statically_known;

const CONST_TEST: bool = unsafe { is_val_statically_known(0) };

fn main() {
    if CONST_TEST {
        unreachable!("currently expected to return false during const eval");
        // but note that this is not a guarantee!
    }
}
