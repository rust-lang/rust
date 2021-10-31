// build-pass
// only-x86_64

#![feature(asm, target_feature_11)]

#[target_feature(enable = "avx")]
fn main() {
    unsafe {
        asm!(
            "/* {} */",
            out(ymm_reg) _,
        );
    }
}
