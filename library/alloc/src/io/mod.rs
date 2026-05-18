//! Traits, helpers, and type definitions for core I/O functionality.

mod error;

#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use core::io::RawOsError;
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use core::io::SimpleMessage;
#[unstable(feature = "io_const_error", issue = "133448")]
pub use core::io::const_error;
#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use core::io::{BorrowedBuf, BorrowedCursor};
#[unstable(feature = "alloc_io", issue = "154046")]
pub use core::io::{
    Chain, Cursor, Empty, Error, ErrorKind, IoSlice, IoSliceMut, Repeat, Result, Sink, Take, empty,
    repeat, sink,
};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use core::io::{
    Custom, CustomOwner, OsFunctions, chain, decode_error_kind, format_os_error, is_interrupted,
    set_functions, take,
};

#[cfg(not(no_global_oom_handling))]
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::error::custom_owner_from_box;
