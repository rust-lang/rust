#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

//error-pattern:literal items (e.g. mentions of function items) are unimplemented

static mut X: usize = 5;

#[miri_run]
fn static_mut() {
    unsafe {
        X = 6;
        assert_eq!(X, 6);
    }
}

fn main() {}
