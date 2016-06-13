#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

fn f() {}

#[miri_run]
fn deref_fn_ptr() -> i32 {
    unsafe {
        *std::mem::transmute::<fn(), *const i32>(f) //~ ERROR: tried to dereference a function pointer
    }
}

fn main() {}
