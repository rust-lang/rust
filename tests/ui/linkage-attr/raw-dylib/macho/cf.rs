//! Link to CoreFoundation without having to have the linker stubs available.

//@ only-apple
//@ run-pass

#![allow(incomplete_features)]
#![feature(raw_dylib_macho)]

#[link(
    name = "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
    kind = "raw-dylib",
    modifiers = "+verbatim"
)]
unsafe extern "C" {
    // Example function.
    safe fn CFRunLoopGetTypeID() -> core::ffi::c_ulong;
}

fn main() {
    let _ = CFRunLoopGetTypeID();
}
