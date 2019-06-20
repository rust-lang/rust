#![feature(core_intrinsics)]

use std::intrinsics::*;

//error-pattern: Division by 0 in unchecked_rem

fn main() {
    unsafe {
        let _n = unchecked_rem(3u32, 0);
    }
}
