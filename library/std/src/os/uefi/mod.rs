//! Platform-specific extensions to `std` for UEFI.

#![unstable(feature = "uefi_std", issue = "100499")]

pub mod env;
pub mod ffi;
pub mod io;
pub mod net;
pub mod raw;
