//@ revisions: arm-mode thumb-mode
//@ [arm-mode] compile-flags: --target armv5te-none-eabi
//@ [thumb-mode] compile-flags: --target thumbv5te-none-eabi
//@ [arm-mode] needs-llvm-components: arm
//@ [thumb-mode] needs-llvm-components: arm

#![crate_type = "lib"]
#![feature(no_core, lang_items, rustc_attrs, naked_functions)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! naked_asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

// arm-mode: .arm
// thumb-mode: .thumb
// CHECK-LABEL: test_unspecified:
// CHECK: bx lr
// CHECK: .popsection
// arm-mode: .arm
// thumb-mode: .thumb
#[no_mangle]
#[naked]
unsafe extern "C" fn test_unspecified() {
    naked_asm!("bx lr");
}

// CHECK: .thumb
// CHECK: .thumb_func
// CHECK-LABEL: test_thumb:
// CHECK: bx lr
// CHECK: .popsection
// arm-mode: .arm
// thumb-mode: .thumb
#[no_mangle]
#[naked]
#[instruction_set(arm::t32)]
unsafe extern "C" fn test_thumb() {
    naked_asm!("bx lr");
}

// CHECK: .arm
// CHECK-LABEL: test_arm:
// CHECK: bx lr
// CHECK: .popsection
// arm-mode: .arm
// thumb-mode: .thumb
#[no_mangle]
#[naked]
#[instruction_set(arm::a32)]
unsafe extern "C" fn test_arm() {
    naked_asm!("bx lr");
}
