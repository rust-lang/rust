// Test that `objc::class!` and `objc::selector!` can't be returned by reference.
// A single instance may have multiple addresses (e.g. across dylib boundaries).

//@ edition: 2024
//@ only-apple

#![feature(darwin_objc)]

use std::os::darwin::objc;

pub fn class_ref<'a>() -> &'a objc::Class {
    &objc::class!("NSObject")
    //~^ ERROR cannot return reference to temporary value [E0515]
}

pub fn class_ref_static() -> &'static objc::Class {
    &objc::class!("NSObject")
    //~^ ERROR cannot return reference to temporary value [E0515]
}

pub fn selector_ref<'a>() -> &'a objc::SEL {
    &objc::selector!("alloc")
    //~^ ERROR cannot return reference to temporary value [E0515]
}

pub fn selector_ref_static() -> &'static objc::SEL {
    &objc::selector!("alloc")
    //~^ ERROR cannot return reference to temporary value [E0515]
}

pub fn main() {}
