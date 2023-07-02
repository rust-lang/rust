//@run-rustfix
//@aux-build:proc_macros.rs:proc-macro

#![warn(clippy::ptr_cast_constness)]
#![allow(clippy::transmute_ptr_to_ref, clippy::unnecessary_cast, unused)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

unsafe fn ptr_to_ref<T, U>(p: *const T, om: *mut U) {
    let _: &mut T = std::mem::transmute(p as *mut T);
    let _ = &mut *(p as *mut T);
    let _: &T = &*(om as *const T);
}

#[inline_macros]
fn main() {
    let ptr: *const u32 = &42_u32;
    let mut_ptr: *mut u32 = &mut 42_u32;

    let _ = ptr as *const u32;
    let _ = mut_ptr as *mut u32;

    // Make sure the lint can handle the difference in their operator precedences.
    unsafe {
        let ptr_ptr: *const *const u32 = &ptr;
        let _ = *ptr_ptr as *mut u32;
    }

    let _ = ptr as *mut u32;
    let _ = mut_ptr as *const u32;

    // Lint this, since pointer::cast_mut and pointer::cast_const have ?Sized
    let ptr_of_array: *const [u32; 4] = &[1, 2, 3, 4];
    let _ = ptr_of_array as *const [u32];
    let _ = ptr_of_array as *const dyn std::fmt::Debug;

    // Make sure the lint is triggered inside a macro
    let _ = inline!($ptr as *const u32);

    // Do not lint inside macros from external crates
    let _ = external!($ptr as *const u32);
}

#[clippy::msrv = "1.64"]
fn _msrv_1_64() {
    let ptr: *const u32 = &42_u32;
    let mut_ptr: *mut u32 = &mut 42_u32;

    // `pointer::cast_const` and `pointer::cast_mut` were stabilized in 1.65. Do not lint this
    let _ = ptr as *mut u32;
    let _ = mut_ptr as *const u32;
}

#[clippy::msrv = "1.65"]
fn _msrv_1_65() {
    let ptr: *const u32 = &42_u32;
    let mut_ptr: *mut u32 = &mut 42_u32;

    let _ = ptr as *mut u32;
    let _ = mut_ptr as *const u32;
}
