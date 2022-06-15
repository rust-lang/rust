//! Owned and borrowed Unix-like file descriptors.

#![stable(feature = "io_safety", since = "1.63.0")]
#![deny(unsafe_op_in_unsafe_fn)]

// `RawFd`, `AsRawFd`, etc.
pub mod raw;

// `OwnedFd`, `AsFd`, etc.
pub mod owned;

// Implementations for `AsRawFd` etc. for network types.
mod net;

#[cfg(test)]
mod tests;
