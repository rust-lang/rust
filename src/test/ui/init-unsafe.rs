#![allow(deprecated)]
#![feature(core_intrinsics)]

use std::intrinsics::{init};

// Test that the `init` intrinsic is really unsafe
pub fn main() {
    let stuff = init::<isize>(); //~ ERROR call to unsafe function is unsafe
}
