//! Traits, helpers, and type definitions for core I/O functionality.

#![unstable(feature = "core_io", issue = "none")]

mod borrowed_buf;

#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use self::borrowed_buf::{BorrowedBuf, BorrowedCursor};

#[cfg(target_has_atomic_load_store = "ptr")]
mod error;

#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use self::error::RawOsError;
#[doc(hidden)]
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use self::error::SimpleMessage;
#[cfg(target_has_atomic_load_store = "ptr")]
#[unstable(feature = "io_const_error", issue = "133448")]
pub use self::error::const_error;
#[unstable(feature = "core_io_error", issue = "none")]
#[cfg(target_has_atomic_load_store = "ptr")]
pub use self::error::{Error, ErrorKind, Result};

#[cfg(target_has_atomic_load_store = "ptr")]
#[doc(hidden)]
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub mod error_internals {
    //! implementation detail of [`Error`]

    pub use super::error::*;
}
