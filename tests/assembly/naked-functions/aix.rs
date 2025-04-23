//@ revisions: elfv1-be aix
//@ add-core-stubs
//@ assembly-output: emit-asm
//
//@[elfv1-be] compile-flags: --target powerpc64-unknown-linux-gnu
//@[elfv1-be] needs-llvm-components: powerpc
//
//@[aix] compile-flags: --target powerpc64-ibm-aix
//@[aix] needs-llvm-components: powerpc

#![crate_type = "lib"]
#![feature(no_core, asm_experimental_arch, f128, linkage, fn_align)]
#![no_core]

// tests that naked functions work for the `powerpc64-ibm-aix` target.
//
// This target is special because it uses the XCOFF binary format
// It is tested alongside an elf powerpc target to pin down commonalities and differences.
//
// https://doc.rust-lang.org/rustc/platform-support/aix.html
// https://www.ibm.com/docs/en/aix/7.2?topic=formats-xcoff-object-file-format

extern crate minicore;
use minicore::*;

// elfv1-be: .p2align 2
// aix: .align 2
// CHECK: .globl blr
// CHECK-LABEL: blr:
// CHECK: blr
#[no_mangle]
#[unsafe(naked)]
extern "C" fn blr() {
    naked_asm!("blr")
}
