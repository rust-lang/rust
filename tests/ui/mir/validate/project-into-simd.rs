// Optimized MIR shouldn't have critical call edges
//
//@ build-fail
//@ edition: 2021
//@ compile-flags: --crate-type=lib
//@ failure-status: 101
//@ dont-check-compiler-stderr

#![feature(repr_simd)]

#[repr(simd)]
pub struct U32x4([u32; 4]);

pub fn f(a: U32x4) -> [u32; 4] {
    a.0
    //~^ ERROR broken MIR in Item
    //~| ERROR Projecting into SIMD type U32x4 is banned by MCP#838
}
