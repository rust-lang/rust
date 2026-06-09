// Separate test without strict provenance
//@compile-flags: -Zmiri-permissive-provenance
#![feature(portable_simd)]
use std::ptr;
use std::simd::prelude::*;

fn main() {
    // Pointer casts
    let _val: Simd<*const u8, 4> = Simd::<*const i32, 4>::splat(ptr::null()).cast();
    let addrs = Simd::<*const i32, 4>::splat(ptr::null()).expose_provenance();
    let _ptrs = Simd::<*const i32, 4>::with_exposed_provenance(addrs);
}
