//! Owned and borrowed Unix-like file descriptors.
//!
//! This module is supported on Unix platforms and WASI, which both use a
//! similar file descriptor system for referencing OS resources.

#![stable(feature = "os_fd", since = "1.66.0")]
#![deny(unsafe_op_in_unsafe_fn)]

// `RawFd`, `AsRawFd`, etc.
mod raw;

// `OwnedFd`, `AsFd`, etc.
mod owned;

// `CommandExt`, etc.
#[cfg(unix)]
#[unstable(feature = "command_pass_fds", issue = "144989")]
pub mod process;

// Implementations for `AsRawFd` etc. for network types.
#[cfg(not(target_os = "trusty"))]
mod net;

#[cfg(test)]
mod tests;

// Export the types and traits for the public API.
#[stable(feature = "os_fd", since = "1.66.0")]
pub use owned::*;
#[stable(feature = "os_fd", since = "1.66.0")]
pub use raw::*;
