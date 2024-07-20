//@ revisions: linux windows macos thumb
//
//@[linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[windows] compile-flags: --target x86_64-pc-windows-gnu
//@[windows] needs-llvm-components: x86
//@[macos] compile-flags: --target aarch64-apple-darwin
//@[macos] needs-llvm-components: arm
//@[thumb] compile-flags: --target thumbv7em-none-eabi
//@[thumb] needs-llvm-components: arm

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

// linux,windows: .intel_syntax
//
// linux:   .pushsection .text.naked_empty,\22ax\22, @progbits
// macos:   .pushsection __TEXT,__text,regular,pure_instructions
// windows: .pushsection .text.naked_empty,\22xr\22
// thumb:   .pushsection .text.naked_empty,\22ax\22, %progbits
//
// CHECK: .balign 4
//
// linux,windows,thumb: .globl naked_empty
// macos: .globl _naked_empty
//
// CHECK-NOT: .private_extern
// CHECK-NOT: .hidden
//
// linux: .type naked_empty, @function
//
// windows: .def naked_empty
// windows: .scl 2
// windows: .type 32
// windows: .endef naked_empty
//
// thumb: .type naked_empty, %function
// thumb: .thumb
// thumb: .thumb_func
//
// CHECK-LABEL: naked_empty:
//
// linux,macos,windows: ret
// thumb: bx lr
//
// CHECK: .popsection
//
// thumb: .thumb
//
// linux,windows: .att_syntax

#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_empty() {
    #[cfg(not(all(target_arch = "arm", target_feature = "thumb-mode")))]
    naked_asm!("ret");

    #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))]
    naked_asm!("bx lr");
}

// linux,windows: .intel_syntax
//
// linux:   .pushsection .text.naked_with_args_and_return,\22ax\22, @progbits
// macos:   .pushsection __TEXT,__text,regular,pure_instructions
// windows: .pushsection .text.naked_with_args_and_return,\22xr\22
// thumb:   .pushsection .text.naked_with_args_and_return,\22ax\22, %progbits
//
// CHECK: .balign 4
//
// linux,windows,thumb: .globl naked_with_args_and_return
// macos: .globl _naked_with_args_and_return
//
// CHECK-NOT: .private_extern
// CHECK-NOT: .hidden
//
// linux: .type naked_with_args_and_return, @function
//
// windows: .def naked_with_args_and_return
// windows: .scl 2
// windows: .type 32
// windows: .endef naked_with_args_and_return
//
// thumb: .type naked_with_args_and_return, %function
// thumb: .thumb
// thumb: .thumb_func
//
// CHECK-LABEL: naked_with_args_and_return:
//
// linux, windows: lea rax, [rdi + rsi]
// macos: add x0, x0, x1
// thumb: adds r0, r0, r1
//
// linux,macos,windows: ret
// thumb: bx lr
//
// CHECK: .popsection
//
// thumb: .thumb
//
// linux,windows: .att_syntax

#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_with_args_and_return(a: isize, b: isize) -> isize {
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        naked_asm!("lea rax, [rdi + rsi]", "ret")
    }

    #[cfg(target_os = "macos")]
    {
        naked_asm!("add x0, x0, x1", "ret")
    }

    #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))]
    {
        naked_asm!("adds r0, r0, r1", "bx lr")
    }
}

// linux:   .pushsection .text.some_different_name,\22ax\22, @progbits
// macos:   .pushsection .text.some_different_name,regular,pure_instructions
// windows: .pushsection .text.some_different_name,\22xr\22
// thumb:   .pushsection .text.some_different_name,\22ax\22, %progbits
// CHECK-LABEL: test_link_section:
#[no_mangle]
#[naked]
#[link_section = ".text.some_different_name"]
pub unsafe extern "C" fn test_link_section() {
    #[cfg(not(all(target_arch = "arm", target_feature = "thumb-mode")))]
    naked_asm!("ret", options(noreturn));

    #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))]
    naked_asm!("bx lr", options(noreturn));
}
