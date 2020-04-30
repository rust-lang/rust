#![feature(core_intrinsics)]

use std::intrinsics;

fn main() {
    unsafe { intrinsics::forget(); } //~ ERROR this function takes 1 argument but 0 arguments were supplied
    unsafe { intrinsics::forget(1, 2); } //~ ERROR this function takes 1 argument but 2 arguments were supplied
}
