// build-pass
// needs-asm-support
// only-x86_64

#![feature(target_feature_11)]

use std::arch::asm;

#[target_feature(enable = "avx")]
fn main() {
    unsafe {
        asm!(
            "/* {} */",
            out(ymm_reg) _,
        );
    }
}
