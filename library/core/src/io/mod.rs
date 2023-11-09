//! Traits, helpers, and type definitions for core I/O functionality.

mod borrowed_buf;

#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use self::borrowed_buf::{BorrowedBuf, BorrowedCursor};
