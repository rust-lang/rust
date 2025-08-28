#![crate_type = "lib"]
#![feature(darwin_objc)]

use std::os::darwin::objc;

#[inline(always)]
pub fn get_class() -> objc::Class {
    objc::class!("MyClass")
}

#[inline(always)]
pub fn get_selector() -> objc::SEL {
    objc::selector!("myMethod")
}
