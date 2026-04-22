//! Traits, helpers, and type definitions for core I/O functionality.

#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use core::io::RawOsError;
#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use core::io::{BorrowedBuf, BorrowedCursor};
#[unstable(feature = "alloc_io", issue = "154046")]
pub use core::io::{
    Chain, Cursor, Empty, ErrorKind, IoSlice, IoSliceMut, Repeat, Sink, Take, empty, repeat, sink,
};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use core::io::{chain, take};
