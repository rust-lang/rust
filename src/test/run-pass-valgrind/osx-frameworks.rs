// no-prefer-dynamic
// pretty-expanded FIXME #23616

#![feature(libc)]

extern crate libc;

#[cfg(target_os = "macos")]
#[link(name = "CoreFoundation", kind = "framework")]
extern {
    fn CFRunLoopGetTypeID() -> libc::c_ulong;
}

#[cfg(target_os = "macos")]
pub fn main() {
    unsafe { CFRunLoopGetTypeID(); }
}

#[cfg(not(target_os = "macos"))]
pub fn main() {}
