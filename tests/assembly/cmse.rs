//@ assembly-output: emit-asm
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib -Copt-level=1
//@ needs-llvm-components: arm
#![crate_type = "lib"]
#![feature(abi_c_cmse_nonsecure_call, cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]
#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}

// CHECK-LABEL: __acle_se_entry_point
// CHECK: bxns
#[no_mangle]
pub extern "C-cmse-nonsecure-entry" fn entry_point() -> i64 {
    0
}

// CHECK-LABEL: call_nonsecure
// CHECK: blxns
#[no_mangle]
pub fn call_nonsecure(
    f: unsafe extern "C-cmse-nonsecure-call" fn(u32, u32, u32, u32) -> u64,
) -> u64 {
    unsafe { f(0, 1, 2, 3) }
}
