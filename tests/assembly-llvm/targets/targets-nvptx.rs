//@ add-core-stubs
//@ assembly-output: emit-asm
// ignore-tidy-linelength
//@ revisions: nvptx64_nvidia_cuda
//@ [nvptx64_nvidia_cuda] compile-flags: --target nvptx64-nvidia-cuda
//@ [nvptx64_nvidia_cuda] needs-llvm-components: nvptx

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
