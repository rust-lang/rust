//@ add-minicore
//@ assembly-output: emit-asm
//@ revisions: spirv_unknown_vulkan1_3
//@ [spirv_unknown_vulkan1_3] compile-flags: --target spirv-unknown-vulkan1.3
//@ [spirv_unknown_vulkan1_3] needs-llvm-components: spirv

// Sanity-check that each target can produce assembly code.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

pub fn test() -> u8 {
    42
}

// CHECK: OpCapability
