//! Unwinding for *hermit* target.
//!
//! Right now we don't support this, so this is just stubs.

use alloc::boxed::Box;
use core::any::Any;

pub(crate) unsafe fn cleanup(_ptr: *mut u8) -> Box<dyn Any + Send> {
    extern "C" {
        fn __rust_abort() -> !;
    }
    __rust_abort();
}

pub(crate) unsafe fn panic(_data: Box<dyn Any + Send>) -> u32 {
    extern "C" {
        fn __rust_abort() -> !;
    }
    __rust_abort();
}
