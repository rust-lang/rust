//! Check that linking to UIKit on platforms where that is available works.
//@ revisions: ios tvos watchos visionos
//@ [ios]only-ios
//@ [tvos]only-tvos
//@ [watchos]only-watchos
//@ [visionos]only-visionos
//@ build-pass

use std::ffi::{c_char, c_int, c_void};

#[link(name = "UIKit", kind = "framework")]
extern "C" {
    pub fn UIApplicationMain(
        argc: c_int,
        argv: *const c_char,
        principalClassName: *const c_void,
        delegateClassName: *const c_void,
    ) -> c_int;
}

pub fn main() {
    unsafe {
        UIApplicationMain(0, core::ptr::null(), core::ptr::null(), core::ptr::null());
    }
}
