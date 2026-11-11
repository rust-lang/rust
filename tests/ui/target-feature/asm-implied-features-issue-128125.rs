//@ only-x86_64
//@ build-pass
#![allow(dead_code)]

#[target_feature(enable = "avx2")]
unsafe fn demo(v: std::arch::x86_64::__m256i) {
    std::arch::asm!("/* {v} */", v = in(ymm_reg) v);
}

fn main() {}
