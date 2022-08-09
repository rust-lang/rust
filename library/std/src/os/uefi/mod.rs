//! Platform-specific extensions to `std` for UEFI.

#![unstable(feature = "uefi_std", issue = "none")]

pub mod env;
pub mod ffi;
pub mod io;
pub mod net;
pub mod path;
pub mod raw;
