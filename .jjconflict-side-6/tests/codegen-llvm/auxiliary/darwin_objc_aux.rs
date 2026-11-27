#![crate_type = "lib"]
#![feature(darwin_objc)]

use std::os::darwin::objc;

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[inline(always)]
pub fn inline_get_object_class() -> objc::Class {
    objc::class!("NSObject")
}

#[inline(always)]
pub fn inline_get_alloc_selector() -> objc::SEL {
    objc::selector!("alloc")
}

#[inline(never)]
pub fn never_inline_get_string_class() -> objc::Class {
    objc::class!("NSString")
}

#[inline(never)]
pub fn never_inline_get_init_selector() -> objc::SEL {
    objc::selector!("init")
}
