//! QuRT-specific I/O functionality.
//!
//! QuRT supports Unix-like file descriptors through its POSIX compatibility layer.

#![stable(feature = "raw_ext", since = "1.1.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::os::fd::*;
