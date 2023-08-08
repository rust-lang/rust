// Separate test without strict provenance
//@compile-flags: -Zmiri-permissive-provenance
#![feature(portable_simd, platform_intrinsics)]
use std::ptr;
use std::simd::*;

fn main() {
    // Pointer casts
    let _val: Simd<*const u8, 4> = Simd::<*const i32, 4>::splat(ptr::null()).cast();
    let addrs = Simd::<*const i32, 4>::splat(ptr::null()).expose_addr();
    let _ptrs = Simd::<*const i32, 4>::from_exposed_addr(addrs);
}
