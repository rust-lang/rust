#![feature(core_intrinsics)]

use std::intrinsics::*;

//error-pattern: Division by 0 in unchecked_div

fn main() {
    unsafe {
        let _n = unchecked_div(1i64, 0);
    }
}
