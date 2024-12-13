//! Unwinding panics for Miri.

use alloc::boxed::Box;
use core::any::Any;

// The type of the payload that the Miri engine propagates through unwinding for us.
// Must be pointer-sized.
type Payload = Box<Box<dyn Any + Send>>;

extern "Rust" {
    /// Miri-provided extern function to begin unwinding.
    fn miri_start_unwind(payload: *mut u8) -> !;
}

pub(crate) unsafe fn panic(payload: Box<dyn Any + Send>) -> u32 {
    // The payload we pass to `miri_start_unwind` will be exactly the argument we get
    // in `cleanup` below. So we just box it up once, to get something pointer-sized.
    let payload_box: Payload = Box::new(payload);
    miri_start_unwind(Box::into_raw(payload_box) as *mut u8)
}

pub(crate) unsafe fn cleanup(payload_box: *mut u8) -> Box<dyn Any + Send> {
    // Recover the underlying `Box`.
    let payload_box: Payload = Box::from_raw(payload_box as *mut _);
    *payload_box
}
