//! Unwinding for *rtems* target.
//!
//! Right now we don't support this, so this is just stubs.

use alloc::boxed::Box;
use core::any::Any;

pub unsafe fn cleanup(_ptr: *mut u8) -> Box<dyn Any + Send> {
    extern "C" {
        pub fn rtems_panic(fmt: *const ::std::os::raw::c_char, ...) -> !;
    }

    rtems_panic("Error during Rust execution".into_raw());
}

pub unsafe fn panic(_data: Box<dyn Any + Send>) -> u32 {
    extern "C" {
        pub fn rtems_panic(fmt: *const ::std::os::raw::c_char, ...) -> !;
    }

    rtems_panic("Error during Rust execution".into_raw());
}
