//@ compile-flags: --target armv5te-unknown-linux-gnueabi
//@ needs-llvm-components: arm
//@ needs-asm-support
//@ build-pass

#![crate_type = "lib"]
#![feature(no_core, lang_items, rustc_attrs, naked_functions)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! naked_asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}

#[no_mangle]
#[naked]
#[instruction_set(arm::t32)]
unsafe extern "C" fn test_thumb() {
    naked_asm!("bx lr");
}

#[no_mangle]
#[naked]
#[instruction_set(arm::t32)]
unsafe extern "C" fn test_arm() {
    naked_asm!("bx lr");
}
