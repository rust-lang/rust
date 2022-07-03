//! Platform-specific extensions to `std` for UEFI.

#![unstable(feature = "uefi_std", issue = "none")]

pub mod env;
pub mod ffi;
pub mod raw;

#[cfg(test)]
mod tests;
