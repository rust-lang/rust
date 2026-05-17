//! Traits, helpers, and type definitions for core I/O functionality.

mod borrowed_buf;
mod cursor;
mod error;
mod util;

#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use self::borrowed_buf::{BorrowedBuf, BorrowedCursor};
#[unstable(feature = "core_io", issue = "154046")]
pub use self::cursor::Cursor;
#[unstable(feature = "core_io", issue = "154046")]
pub use self::error::ErrorKind;
#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use self::error::RawOsError;
#[unstable(feature = "core_io", issue = "154046")]
pub use self::util::{Chain, Empty, Repeat, Sink, Take, empty, repeat, sink};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::util::{chain, take};
