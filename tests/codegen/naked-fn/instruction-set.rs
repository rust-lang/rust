//@ compile-flags: --target armv5te-none-eabi
//@ needs-llvm-components: arm

#![crate_type = "lib"]
#![feature(no_core, lang_items, rustc_attrs, naked_functions)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

// CHECK-LABEL: test_unspecified:
// CHECK: .arm
#[no_mangle]
#[naked]
unsafe extern "C" fn test_unspecified() {
    asm!("bx lr", options(noreturn));
}

// CHECK-LABEL: test_thumb:
// CHECK: .thumb
// CHECK: .thumb_func
#[no_mangle]
#[naked]
#[instruction_set(arm::t32)]
unsafe extern "C" fn test_thumb() {
    asm!("bx lr", options(noreturn));
}

// CHECK-LABEL: test_arm:
// CHECK: .arm
#[no_mangle]
#[naked]
#[instruction_set(arm::t32)]
unsafe extern "C" fn test_arm() {
    asm!("bx lr", options(noreturn));
}
