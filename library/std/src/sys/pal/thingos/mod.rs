#![deny(unsafe_op_in_unsafe_fn)]

// ThingOS targets are currently abort-only for panics.
// Emit a clear compile-time error for any attempt to build std with unwind.
#[cfg(panic = "unwind")]
compile_error!(
    "ThingOS std PAL is abort-only: panic=unwind is not supported for target_os=thingos"
);

pub mod common;
pub mod futex;
pub mod os;
pub mod time;

pub use common::*;
