// run-pass
#![feature(core_intrinsics)]

use std::intrinsics::*;

fn main() {
    unsafe {
        assert_eq!(nearbyintf32(5.234f32), 5f32);
        assert_eq!(nearbyintf64(6.777f64), 7f64);
    }
}
