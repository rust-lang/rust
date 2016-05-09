#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

// This tests that the size of Option<Box<i32>> is the same as *const i32.

#[miri_run]
fn option_box_deref() -> i32 {
    let val = Some(Box::new(42));
    unsafe {
        let ptr: *const i32 = std::mem::transmute::<Option<Box<i32>>, *const i32>(val);
        *ptr
    }
}

fn main() {}
