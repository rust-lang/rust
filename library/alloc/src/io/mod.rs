//! Traits, helpers, and type definitions for core I/O functionality.

#[unstable(feature = "read_buf", issue = "78485")]
pub use core::io::{BorrowedBuf, BorrowedCursor};

#[doc(hidden)]
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use self::error::SimpleMessage;
#[unstable(feature = "io_const_error", issue = "133448")]
pub use self::error::const_error;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::error::{Error, ErrorKind, Result};
#[unstable(feature = "io_error_internals", issue = "none")]
#[doc(hidden)]
#[cfg(target_has_atomic = "ptr")]
pub use self::error::{RawOsError, os::OsFunctions};

mod error;
