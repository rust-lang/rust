//! Traits, helpers, and type definitions for core I/O functionality.

mod borrowed_buf;
mod cursor;
mod error;
mod io_slice;
mod util;

#[unstable(feature = "core_io_borrowed_buf", issue = "117693")]
pub use self::borrowed_buf::{BorrowedBuf, BorrowedCursor};
#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use self::error::RawOsError;
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use self::error::SimpleMessage;
#[unstable(feature = "io_const_error", issue = "133448")]
pub use self::error::const_error;
#[unstable(feature = "core_io", issue = "154046")]
pub use self::{
    cursor::Cursor,
    error::{Error, ErrorKind, Result},
    io_slice::{IoSlice, IoSliceMut},
    util::{Chain, Empty, Repeat, Sink, Take, empty, repeat, sink},
};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::{
    error::{
        Custom, CustomOwner, OsFunctions, decode_error_kind, format_os_error, is_interrupted,
        set_functions,
    },
    util::{chain, take},
};
