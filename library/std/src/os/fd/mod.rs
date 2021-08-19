//! Owned and borrowed Unix-like file descriptors.

#![unstable(feature = "io_safety", issue = "87074")]
#![deny(unsafe_op_in_unsafe_fn)]

// `RawFd`, `AsRawFd`, etc.
pub mod raw;

// `OwnedFd`, `AsFd`, etc.
pub mod owned;

// Implementations for `AsRawFd` etc. for network types.
mod net;
