//@ add-core-stubs
//@ assembly-output: emit-asm
//@ needs-llvm-components: x86
//@ revisions: TWOFLAGS SINGLEFLAG
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ [TWOFLAGS] compile-flags: -C target-feature=+rdrnd -C target-feature=+rdseed
//@ [SINGLEFLAG] compile-flags: -C target-feature=+rdrnd,+rdseed

// Target features set via flags aren't necessarily reflected in the IR, so the only way to test
// them is to build code that requires the features to be enabled to work.
//
// In this particular test if `rdrnd,rdseed` somehow didn't make it to LLVM, the instruction
// selection should crash.
//
// > LLVM ERROR: Cannot select: 0x7f00f400c010: i32,i32,ch = X86ISD::RDSEED 0x7f00f400bfa8:2
// > In function: foo
//
// See also tests/codegen-llvm/target-feature-overrides.rs
#![feature(no_core, lang_items, link_llvm_intrinsics, abi_unadjusted)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// Use of these requires target features to be enabled
extern "unadjusted" {
    #[link_name = "llvm.x86.rdrand.32"]
    fn x86_rdrand32_step() -> (u32, i32);
    #[link_name = "llvm.x86.rdseed.32"]
    fn x86_rdseed32_step() -> (u32, i32);
}

#[no_mangle]
pub unsafe fn foo() -> (u32, u32) {
    // CHECK-LABEL: foo:
    // CHECK: rdrand
    // CHECK: rdseed
    (x86_rdrand32_step().0, x86_rdseed32_step().0)
}
