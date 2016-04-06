#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn overwriting_part_of_relocation_makes_the_rest_undefined() -> i32 {
    let mut p: *const i32 = &42;
    unsafe {
        let ptr = &mut p as *mut *const i32 as *mut u32;
        *ptr = 123;
        *p
    }
}
