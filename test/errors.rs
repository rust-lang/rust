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
