// only-aarch64
// run-pass
use std::arch::aarch64::*;

// Smoke test to verify aarch64 code that enables NEON compiles.
fn main() {
    let zero = if is_aarch64_feature_detected!("neon") {
        unsafe {
            let zeros = zero_vector();
            vget_lane_u8::<1>(1)
        }
    } else {
        0
    };
    if cfg!(target feature = "neon") {
        assert_eq!(zero, 0)
    };
}


#[target_feature(enable = "neon")]
unsafe fn zero_vector() -> uint8x16_t {
    vmovq_n_u8(0)
}
