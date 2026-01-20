//@ add-minicore
//@ assembly-output: emit-asm
//@ revisions: spirv64_intel_unknown
//@ [spirv64_intel_unknown] compile-flags: --target spirv64-intel-unknown
//@ [spirv64_intel_unknown] needs-llvm-components: spirv

// Sanity-check that each target can produce assembly code.

#![feature(no_core, lang_items, never_type)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

pub fn test() -> u8 {
    42
}

// CHECK: OpCapability Kernel
