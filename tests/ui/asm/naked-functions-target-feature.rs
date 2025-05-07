//@ build-pass
//@ needs-asm-support

#![feature(naked_functions_target_feature)]
#![crate_type = "lib"]

use std::arch::{asm, naked_asm};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[unsafe(naked)]
pub extern "C" fn compatible_target_feature() {
    naked_asm!("ret");
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[unsafe(naked)]
pub extern "C" fn compatible_target_feature() {
    naked_asm!("ret");
}
