#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn option_box_deref() -> i32 {
    let val = Some(Box::new(42));
    unsafe {
        let ptr: *const i32 = std::mem::transmute(val); //~ ERROR: pointer offset outside bounds of allocation
        *ptr
    }
}

fn main() {}
