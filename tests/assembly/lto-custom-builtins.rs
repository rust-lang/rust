// assembly-output: emit-asm
// compile-flags: --crate-type cdylib -C lto=fat -C prefer-dynamic=no
// only-x86_64-unknown-linux-gnu

#![feature(lang_items, linkage)]
#![no_std]
#![no_main]

#![crate_type = "bin"]

// We want to use customized __subdf3.
// CHECK: .globl __subdf3
// CHECK-NEXT: __subdf3:
// CHECK-NEXT: movq $2, %rax
core::arch::global_asm!(".global __subdf3", "__subdf3:", "mov rax, 2");

// We want to use customized __addsf3.
// CHECK: .globl __addsf3
// CHECK: __addsf3:
// CHECK: xorl %eax, %eax
// CHECK-NEXT retq
#[no_mangle]
pub extern "C" fn __addsf3() -> i32 {
    0
}

// We want to use __adddf3 of compiler-builtins.
// CHECK: .globl __adddf3
// CHECK: __adddf3:
// CHECK-NEXT: .cfi_startproc
// CHECK-NOT: movl $1, %eax
// CHECK: movq %xmm0, %rdx
#[no_mangle]
#[linkage = "weak"]
pub extern "C" fn __adddf3() -> i32 {
    1
}

#[panic_handler]
fn panic(_panic: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
