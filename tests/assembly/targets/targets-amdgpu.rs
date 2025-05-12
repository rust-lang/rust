//@ add-core-stubs
//@ assembly-output: emit-asm
// ignore-tidy-linelength
//@ revisions: amdgcn_amd_amdhsa
//@ [amdgcn_amd_amdhsa] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [amdgcn_amd_amdhsa] needs-llvm-components: amdgpu

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

// CHECK: .version
