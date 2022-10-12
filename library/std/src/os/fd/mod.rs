//! Owned and borrowed Unix-like file descriptors.
//!
//! This module is supported on Unix platforms and WASI, which both use a
//! similar file descriptor system for referencing OS resources.

#![stable(feature = "io_safety", since = "1.63.0")]
#![deny(unsafe_op_in_unsafe_fn)]

// `RawFd`, `AsRawFd`, etc.
mod raw;

// `OwnedFd`, `AsFd`, etc.
mod owned;

// Implementations for `AsRawFd` etc. for network types.
mod net;

#[cfg(test)]
mod tests;

// Export the types and traits for the public API.
#[unstable(feature = "os_fd", issue = "98699")]
pub use owned::*;
#[unstable(feature = "os_fd", issue = "98699")]
pub use raw::*;
