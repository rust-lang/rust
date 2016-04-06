#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn overwriting_part_of_relocation_makes_the_rest_undefined() -> i32 {
    let mut p = &42;
    unsafe {
        let ptr: *mut _ = &mut p;
        *(ptr as *mut u32) = 123;
    }
    *p
}

#[miri_run]
fn pointers_to_different_allocations_are_unorderable() -> bool {
    let x: *const u8 = &1;
    let y: *const u8 = &2;
    x < y
}

#[miri_run]
fn invalid_bools_are_rejected() -> u8 {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) };
    if b { 1 } else { 2 }
}

#[miri_run]
fn undefined_byte_reads_are_rejected() -> u8 {
    let v: Vec<u8> = Vec::with_capacity(10);
    let undef = unsafe { *v.get_unchecked(5) };
    undef + 1
}

#[miri_run]
fn out_of_bounds_reads_are_rejected() -> u8 {
    let v: Vec<u8> = vec![1, 2];
    unsafe { *v.get_unchecked(5) }
}
