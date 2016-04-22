#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn overwriting_part_of_relocation_makes_the_rest_undefined() -> i32 {
    let mut p = &42;
    unsafe {
        let ptr: *mut _ = &mut p;
        *(ptr as *mut u32) = 123;
    }
    *p //~ ERROR: attempted to read undefined bytes
}

#[miri_run]
fn pointers_to_different_allocations_are_unorderable() -> bool {
    let x: *const u8 = &1;
    let y: *const u8 = &2;
    x < y //~ ERROR: attempted to do math or a comparison on pointers into different allocations
}

#[miri_run]
fn invalid_bool() -> u8 {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) };
    if b { 1 } else { 2 } //~ ERROR: invalid boolean value read
}

#[miri_run]
fn undefined_byte_read() -> u8 {
    let v: Vec<u8> = Vec::with_capacity(10);
    let undef = unsafe { *v.get_unchecked(5) };
    undef + 1 //~ ERROR: attempted to read undefined bytes
}

#[miri_run]
fn out_of_bounds_read() -> u8 {
    let v: Vec<u8> = vec![1, 2];
    unsafe { *v.get_unchecked(5) } //~ ERROR: pointer offset outside bounds of allocation
}

#[miri_run]
fn dangling_pointer_deref() -> i32 {
    let p = {
        let b = Box::new(42);
        &*b as *const i32
    };
    unsafe { *p } //~ ERROR: dangling pointer was dereferenced
}

#[miri_run]
fn wild_pointer_deref() -> i32 {
    let p = 42 as *const i32;
    unsafe { *p } //~ ERROR: attempted to interpret some raw bytes as a pointer address
}

#[miri_run]
fn null_pointer_deref() -> i32 {
    unsafe { *std::ptr::null() } //~ ERROR: attempted to interpret some raw bytes as a pointer address
}

fn main() {}
