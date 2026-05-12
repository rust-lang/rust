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
    error::{Custom, CustomOwner, OsFunctions},
    util::{chain, take},
};

/// Marks that a type `T` can have IO traits such as [`Seek`], [`Write`], etc. automatically
/// implemented for handle types like [`Arc`][arc] as well.
///
/// This trait should only be implemented for types where `<&T as Trait>::method(&mut &value, ..)`
/// would be identical to `<T as Trait>::method(&mut value, ..)`.
///
/// [`File`][file] passes this test, as operations on `&File` and `File` both affect
/// the same underlying file.
/// `[u8]` fails, because any modification to `&mut &[u8]` would only affect a temporary
/// and be lost after the method has been called.
///
/// [file]: ../../std/fs/struct.File.html
/// [arc]: ../../alloc/sync/struct.Arc.html
/// [`Write`]: ../../std/io/trait.Write.html
/// [`Seek`]: ../../std/io/trait.Seek.html
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub trait IoHandle {}
