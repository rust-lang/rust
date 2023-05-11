// pretty-expanded FIXME #23616

#![feature(rustc_private)]

extern crate libc;

#[cfg(target_os = "macos")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRunLoopGetTypeID() -> libc::c_ulong;
}

#[cfg(target_os = "macos")]
pub fn main() {
    unsafe {
        CFRunLoopGetTypeID();
    }
}

#[cfg(not(target_os = "macos"))]
pub fn main() {}
