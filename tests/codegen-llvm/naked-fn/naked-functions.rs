//@ add-core-stubs
//@ revisions: linux win_x86 win_i686 macos thumb
//
//@[linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[win_x86] compile-flags: --target x86_64-pc-windows-gnu
//@[win_x86] needs-llvm-components: x86
//@[win_i686] compile-flags: --target i686-pc-windows-gnu
//@[win_i686] needs-llvm-components: x86
//@[macos] compile-flags: --target aarch64-apple-darwin
//@[macos] needs-llvm-components: aarch64
//@[thumb] compile-flags: --target thumbv7em-none-eabi
//@[thumb] needs-llvm-components: arm

#![crate_type = "lib"]
#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]

extern crate minicore;
use minicore::*;

// linux,win: .intel_syntax
//
// linux:    .pushsection .text.naked_empty,\22ax\22, @progbits
// macos:    .pushsection __TEXT,__text,regular,pure_instructions
// win_x86:  .pushsection .text.naked_empty,\22xr\22
// win_i686: .pushsection .text._naked_empty,\22xr\22
// thumb:    .pushsection .text.naked_empty,\22ax\22, %progbits
//
// CHECK: .balign 4
//
// linux,win,thumb: .globl naked_empty
// macos: .globl _naked_empty
//
// CHECK-NOT: .private_extern
// CHECK-NOT: .hidden
//
// linux: .type naked_empty, @function
//
// win_x86:  .def naked_empty
// win_i686: .def _naked_empty
//
// win_x86,win_i686: .scl 2
// win_x86,win_i686: .type 32
// win_x86,win_i686: .endef
//
// thumb: .type naked_empty, %function
// thumb: .thumb
// thumb: .thumb_func
//
// CHECK-LABEL: naked_empty:
//
// linux,macos,win: ret
// thumb: bx lr
//
// CHECK: .popsection
//
// thumb: .thumb
//
// linux,win: .att_syntax

#[no_mangle]
#[unsafe(naked)]
pub extern "C" fn naked_empty() {
    #[cfg(not(all(target_arch = "arm", target_feature = "thumb-mode")))]
    naked_asm!("ret");

    #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))]
    naked_asm!("bx lr");
}

// linux,win: .intel_syntax
//
// linux:    .pushsection .text.naked_with_args_and_return,\22ax\22, @progbits
// macos:    .pushsection __TEXT,__text,regular,pure_instructions
// win_x86:  .pushsection .text.naked_with_args_and_return,\22xr\22
// win_i686: .pushsection .text._naked_with_args_and_return,\22xr\22
// thumb:    .pushsection .text.naked_with_args_and_return,\22ax\22, %progbits
//
// CHECK: .balign 4
//
// linux,win,thumb: .globl naked_with_args_and_return
// macos: .globl _naked_with_args_and_return
//
// CHECK-NOT: .private_extern
// CHECK-NOT: .hidden
//
// linux: .type naked_with_args_and_return, @function
//
// win_x86:  .def naked_with_args_and_return
// win_i686: .def _naked_with_args_and_return
//
// win_x86,win_i686: .scl 2
// win_x86,win_i686: .type 32
// win_x86,win_i686: .endef
//
// thumb: .type naked_with_args_and_return, %function
// thumb: .thumb
// thumb: .thumb_func
//
// CHECK-LABEL: naked_with_args_and_return:
//
// linux, win_x86,win_i686: lea rax, [rdi + rsi]
// macos: add x0, x0, x1
// thumb: adds r0, r0, r1
//
// linux,macos,win: ret
// thumb: bx lr
//
// CHECK: .popsection
//
// thumb: .thumb
//
// linux,win: .att_syntax

#[no_mangle]
#[unsafe(naked)]
pub extern "C" fn naked_with_args_and_return(a: isize, b: isize) -> isize {
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

// linux:            .pushsection .text.some_different_name,\22ax\22, @progbits
// macos:            .pushsection .text.some_different_name,regular,pure_instructions
// win_x86,win_i686: .pushsection .text.some_different_name,\22xr\22
// thumb:            .pushsection .text.some_different_name,\22ax\22, %progbits
// CHECK-LABEL: test_link_section:
#[no_mangle]
#[unsafe(naked)]
#[link_section = ".text.some_different_name"]
pub extern "C" fn test_link_section() {
    #[cfg(not(all(target_arch = "arm", target_feature = "thumb-mode")))]
    naked_asm!("ret");

    #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))]
    naked_asm!("bx lr");
}

// win_x86:  .def fastcall_cc
// win_i686: .def @fastcall_cc@4
//
// win_x86,win_i686: .scl 2
// win_x86,win_i686: .type 32
// win_x86,win_i686: .endef
//
// win_x86-LABEL: fastcall_cc:
// win_i686-LABEL: @fastcall_cc@4:
#[cfg(target_os = "windows")]
#[no_mangle]
#[unsafe(naked)]
pub extern "fastcall" fn fastcall_cc(x: i32) -> i32 {
    naked_asm!("ret");
}
