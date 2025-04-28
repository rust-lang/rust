//@aux-build:proc_macros.rs

#![warn(clippy::ptr_cast_constness)]
#![allow(
    clippy::transmute_ptr_to_ref,
    clippy::unnecessary_cast,
    unused,
    clippy::missing_transmute_annotations
)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

unsafe fn ptr_to_ref<T, U>(p: *const T, om: *mut U) {
    unsafe {
        let _: &mut T = std::mem::transmute(p as *mut T);
        //~^ ptr_cast_constness
        let _ = &mut *(p as *mut T);
        //~^ ptr_cast_constness
        let _: &T = &*(om as *const T);
    }
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
        //~^ ptr_cast_constness
    }

    let _ = ptr as *mut u32;
    //~^ ptr_cast_constness
    let _ = mut_ptr as *const u32;
    //~^ ptr_cast_constness

    // Lint this, since pointer::cast_mut and pointer::cast_const have ?Sized
    let ptr_of_array: *const [u32; 4] = &[1, 2, 3, 4];
    let _ = ptr_of_array as *const [u32];
    let _ = ptr_of_array as *const dyn std::fmt::Debug;

    // Make sure the lint is triggered inside a macro
    let _ = inline!($ptr as *const u32);

    // Do not lint inside macros from external crates
    let _ = external!($ptr as *const u32);
}

fn lifetime_to_static(v: *mut &()) -> *const &'static () {
    v as _
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
    //~^ ptr_cast_constness
    let _ = mut_ptr as *const u32;
    //~^ ptr_cast_constness
}

#[inline_macros]
fn null_pointers() {
    use std::ptr;
    let _ = ptr::null::<String>() as *mut String;
    //~^ ptr_cast_constness
    let _ = ptr::null_mut::<u32>() as *const u32;
    //~^ ptr_cast_constness
    let _ = ptr::null::<u32>().cast_mut();
    //~^ ptr_cast_constness
    let _ = ptr::null_mut::<u32>().cast_const();
    //~^ ptr_cast_constness

    // Make sure the lint is triggered inside a macro
    let _ = inline!(ptr::null::<u32>() as *mut u32);
    //~^ ptr_cast_constness
    let _ = inline!(ptr::null::<u32>().cast_mut());
    //~^ ptr_cast_constness

    // Do not lint inside macros from external crates
    let _ = external!(ptr::null::<u32>() as *mut u32);
    let _ = external!(ptr::null::<u32>().cast_mut());
}

fn issue14621() {
    let mut local = 4;
    let _ = std::ptr::addr_of_mut!(local) as *const _;
    //~^ ptr_cast_constness
}
