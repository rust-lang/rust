//! ThingOS-specific extensions for file descriptors.

#![stable(feature = "os_thingos", since = "1.0.0")]

#[stable(feature = "os_thingos", since = "1.0.0")]
pub use crate::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
