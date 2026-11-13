//! Unwinding for unsupported target.
//!
//! Stubs that simply abort for targets that don't support unwinding otherwise.

use alloc::boxed::Box;
use alloc::panicking::PanicPayload;
use core::any::Any;

unsafe extern "Rust" {
    // This is defined in std::rt
    #[rustc_std_internal_symbol]
    safe fn __rust_abort() -> !;
}

pub(crate) unsafe fn cleanup(_ptr: *mut u8) -> Box<dyn Any + Send> {
    __rust_abort()
}

pub(crate) fn panic(_data: &mut dyn PanicPayload) -> u32 {
    __rust_abort()
}
