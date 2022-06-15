//! Platform-specific extensions to `std` for UEFI.

#![unstable(feature = "uefi_std", issue = "none")]

pub mod env;

#[cfg(test)]
mod tests;
