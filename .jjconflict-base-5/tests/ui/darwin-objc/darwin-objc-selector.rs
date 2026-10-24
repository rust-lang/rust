// Test that `objc::selector!` returns the same thing as `sel_registerName`.

//@ edition: 2024
//@ only-apple
//@ run-pass

#![feature(darwin_objc)]

use std::ffi::c_char;
use std::os::darwin::objc;

#[link(name = "objc")]
unsafe extern "C" {
    fn sel_registerName(methname: *const c_char) -> objc::SEL;
}

fn get_alloc_selector() -> objc::SEL {
    objc::selector!("alloc")
}

fn register_alloc_selector() -> objc::SEL {
    unsafe { sel_registerName(c"alloc".as_ptr()) }
}

fn get_init_selector() -> objc::SEL {
    objc::selector!("initWithCString:encoding:")
}

fn register_init_selector() -> objc::SEL {
    unsafe { sel_registerName(c"initWithCString:encoding:".as_ptr()) }
}

fn main() {
    assert_eq!(get_alloc_selector(), register_alloc_selector());
    assert_eq!(get_init_selector(), register_init_selector());
}
