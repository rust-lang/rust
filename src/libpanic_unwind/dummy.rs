//! Unwinding for wasm32
//!
//! Right now we don't support this, so this is just stubs

use alloc::boxed::Box;
use core::any::Any;
use core::intrinsics;

pub fn payload() -> *mut u8 {
    0 as *mut u8
}

pub unsafe fn cleanup(_ptr: *mut u8) -> Box<dyn Any + Send> {
    intrinsics::abort()
}

pub unsafe fn panic(_data: Box<dyn Any + Send>) -> u32 {
    intrinsics::abort()
}
