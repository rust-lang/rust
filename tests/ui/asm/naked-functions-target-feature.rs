//@ build-pass
//@ needs-asm-support

#![feature(naked_functions, naked_functions_target_feature)]
#![crate_type = "lib"]

use std::arch::{asm, naked_asm};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[naked]
pub unsafe extern "C" fn compatible_target_feature() {
    naked_asm!("");
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[naked]
pub unsafe extern "C" fn compatible_target_feature() {
    naked_asm!("");
}
