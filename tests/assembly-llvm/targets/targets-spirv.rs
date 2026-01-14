//   @ add-minicore
//@ assembly-output: emit-asm
//@ revisions: spirv_unknown_vulkan1_3
//@ [spirv_unknown_vulkan1_3] compile-flags: --target spirv-unknown-vulkan1.3
//@ [spirv_unknown_vulkan1_3] needs-llvm-components: spirv

// Sanity-check that each target can produce assembly code.

#![feature(no_core, lang_items, never_type)]
#![no_std]
// #![no_core]
#![crate_type = "lib"]

pub enum A {
    Foo(u8),
    Bar(u32),
}

// extern crate minicore;
// use minicore::*;
#[unsafe(no_mangle)]
pub fn test(x: &mut A) -> u8 {
    match x {
        A::Foo(x) => *x,
        A::Bar(b) => *b as u8 + 2,
    }
}

// CHECK: what
