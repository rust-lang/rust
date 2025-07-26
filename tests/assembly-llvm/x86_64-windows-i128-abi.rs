//@ assembly-output: emit-asm
//@ add-core-stubs
//@ revisions: msvc softfloat
//@ compile-flags: -Copt-level=3
//@[msvc] compile-flags: --target x86_64-pc-windows-msvc
//@[msvc] needs-llvm-components: x86
//@[softfloat] compile-flags: --target x86_64-unknown-uefi
//@[softfloat] needs-llvm-components: x86

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: ret_i128
// Hardfloat targets return via xmm0, softfloat targets via rax and rdx.
// msvc: movaps {{.*}}, %xmm0
// softfloat: movq (%[[INPUT:.*]]), %rax
// softfloat-NEXT: movq 8(%[[INPUT]]), %rdx
// CHECK-NEXT: retq
#[no_mangle]
pub extern "C" fn ret_i128(x: &i128) -> i128 {
    *x
}
