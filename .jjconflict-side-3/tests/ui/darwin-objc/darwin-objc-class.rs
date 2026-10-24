// Test that `objc::class!` returns the same thing as `objc_lookUpClass`.

//@ edition: 2024
//@ only-apple
//@ run-pass

#![feature(darwin_objc)]

use std::ffi::c_char;
use std::os::darwin::objc;

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_lookUpClass(methname: *const c_char) -> objc::Class;
}

fn get_object_class() -> objc::Class {
    objc::class!("NSObject")
}

fn lookup_object_class() -> objc::Class {
    unsafe { objc_lookUpClass(c"NSObject".as_ptr()) }
}

fn get_string_class() -> objc::Class {
    objc::class!("NSString")
}

fn lookup_string_class() -> objc::Class {
    unsafe { objc_lookUpClass(c"NSString".as_ptr()) }
}

fn main() {
    assert_eq!(get_object_class(), lookup_object_class());
    assert_eq!(get_string_class(), lookup_string_class());
}
