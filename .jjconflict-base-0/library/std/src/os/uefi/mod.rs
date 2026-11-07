//! Platform-specific extensions to `std` for UEFI.

#![unstable(feature = "uefi_std", issue = "100499")]
#![doc(cfg(target_os = "uefi"))]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod env;
#[path = "../windows/ffi.rs"]
pub mod ffi;
