//! Traits, helpers, and type definitions for core I/O functionality.

mod buf_read;
mod cursor;
mod error;
mod impls;
mod read;
mod util;

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
    Chain, Cursor, Empty, Error, ErrorKind, IoSlice, IoSliceMut, Repeat, Result, Seek, SeekFrom,
    Sink, Take, Write, empty, repeat, sink,
};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use core::io::{
    IoHandle, OsFunctions, SizeHint, WriteThroughCursor, chain, default_write_vectored,
    slice_write, slice_write_all, slice_write_all_vectored, slice_write_vectored,
    stream_len_default, take,
};

#[unstable(feature = "alloc_io", issue = "154046")]
pub use self::{
    buf_read::BufRead,
    read::{Read, read_to_string},
    util::{Bytes, Lines, Split},
};
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::{
    read::{
        DEFAULT_BUF_SIZE, append_to_string, default_read_buf, default_read_buf_exact,
        default_read_exact, default_read_to_end, default_read_to_string, default_read_vectored,
    },
    util::{SpecReadByte, bytes, lines, split, uninlined_slow_read_byte},
};
