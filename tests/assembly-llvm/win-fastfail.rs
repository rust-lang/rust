//@ assembly-output: emit-asm
//@ only-windows

//@ revisions: i686 x86_64 aarch64 arm64ec
//@[i686] only-x86
//@[x86_64] only-x86_64
//@[aarch64] only-aarch64
//@[arm64ec] only-arm64ec

#![feature(abort_immediate)]
#![no_std]
#![crate_type = "lib"]

// CHECK-LABEL: my_abort:
#[unsafe(no_mangle)]
extern "C" fn my_abort() {
    // i686: movl $7, %ecx
    // i686: int $41

    // x86_64: movl $7, %ecx
    // x86_64: int $41

    // aarch64: mov w0, #7
    // aarch64: brk #0xf003

    // arm64ec: mov w0, #7
    // arm64ec: brk #0xf003
    core::process::crash()
}
