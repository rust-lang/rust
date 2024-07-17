//@ compile-flags: --target armv5te-unknown-linux-gnueabi
//@ needs-llvm-components: arm
//@ needs-asm-support
//@ build-pass

#![crate_type = "lib"]
#![feature(no_core, lang_items, rustc_attrs, naked_functions)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}

#[no_mangle]
#[naked]
#[instruction_set(arm::t32)]
unsafe extern "C" fn test_thumb() {
    asm!("bx lr", options(noreturn));
}

#[no_mangle]
#[naked]
#[instruction_set(arm::t32)]
unsafe extern "C" fn test_arm() {
    asm!("bx lr", options(noreturn));
}
