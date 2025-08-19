//! Unwinding for *hermit* target.
//!
//! Right now we don't support this, so this is just stubs.

use alloc::boxed::Box;
use core::any::Any;

unsafe extern "Rust" {
    // This is defined in std::rt
    #[rustc_std_internal_symbol]
    safe fn __rust_abort() -> !;
}

pub(crate) unsafe fn cleanup(_ptr: *mut u8) -> Box<dyn Any + Send> {
    __rust_abort()
}

pub(crate) unsafe fn panic(_data: Box<dyn Any + Send>) -> u32 {
    __rust_abort()
}
