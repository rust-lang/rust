//! Traits, helpers, and type definitions for core I/O functionality.

mod borrowed_buf;
mod error;

#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use self::borrowed_buf::{BorrowedBuf, BorrowedCursor};
#[unstable(feature = "io_error_kind_in_core", issue = "154046")]
pub use self::error::ErrorKind;
