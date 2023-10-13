//! implementation detail of [`Error`]

#[cfg(all(target_pointer_width = "64", not(target_os = "uefi")))]
mod repr_bitpacked;
#[cfg(all(target_pointer_width = "64", bootstrap))]
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub use repr_bitpacked::kind_from_prim;
#[cfg(all(target_pointer_width = "64", not(target_os = "uefi")))]
use repr_bitpacked::Repr;

#[cfg(any(not(target_pointer_width = "64"), target_os = "uefi"))]
mod repr_unpacked;
#[cfg(any(not(target_pointer_width = "64"), target_os = "uefi"))]
use repr_unpacked::Repr;

use crate::alloc::Layout;
use crate::error;
use crate::fmt;
use crate::mem;
use crate::mem::ManuallyDrop;
use crate::ops::Deref;
use crate::ops::DerefMut;
use crate::ptr::NonNull;
use crate::ptr::{self, Unique};
use crate::result;
use crate::str;
use crate::sync::atomic::{AtomicPtr, Ordering};

/// implementation detail of [`Error`]
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub struct ErrorString {
    bytes: ErrorBox<[mem::MaybeUninit<u8>]>,
    length: usize,
}

impl ErrorString {
    /// implementation detail of [`Error`]
    ///
    /// Safety: `(*bytes)[..length]` must be initialized and valid UTF-8
    pub unsafe fn from_raw_parts(bytes: ErrorBox<[mem::MaybeUninit<u8>]>, length: usize) -> Self {
        Self { bytes, length }
    }
}

impl Deref for ErrorString {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        // SAFETY: guaranteed safe by ErrorString's safety invariant
        unsafe {
            str::from_utf8_unchecked(mem::MaybeUninit::slice_assume_init_ref(
                &(*self.bytes)[..self.length],
            ))
        }
    }
}

impl fmt::Debug for ErrorString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl fmt::Display for ErrorString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

/// implementation detail of [`Error`]
///
/// Safety: `self.0` must point to allocated memory and
/// [`AllocVTable::install`] must have been run
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub struct ErrorBox<T: ?Sized>(Unique<T>);

impl<T: ?Sized + fmt::Debug> fmt::Debug for ErrorBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        T::fmt(self, f)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for ErrorBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        T::fmt(self, f)
    }
}

impl<T: ?Sized> Deref for ErrorBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: guaranteed safe by ErrorBox's safety invariant
        unsafe { self.0.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for ErrorBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: guaranteed safe by ErrorBox's safety invariant
        unsafe { self.0.as_mut() }
    }
}

impl<T: ?Sized> ErrorBox<T> {
    /// implementation detail of [`Error`]
    ///
    /// Safety: `v` must point to allocated memory and
    /// [`AllocVTable::install`] must have been run
    pub unsafe fn from_raw(v: *mut T) -> Self {
        // SAFETY: guaranteed by caller
        unsafe { Self(Unique::new_unchecked(v)) }
    }
    /// implementation detail of [`Error`]
    pub fn into_raw(self) -> *mut T {
        ManuallyDrop::new(self).0.as_ptr()
    }
    /// implementation detail of [`Error`]
    pub fn into_inner(self) -> T
    where
        T: Sized,
    {
        let mut b = ManuallyDrop::new(self);
        // SAFETY: guaranteed safe by ErrorBox's safety invariant
        unsafe {
            let _mem = ErrorBoxMem::new(&mut b);
            ptr::read(b.0.as_ptr())
        }
    }
}

struct ErrorBoxMem {
    layout: Layout,
    ptr: NonNull<u8>,
}

impl ErrorBoxMem {
    /// Safety: `b` must not have had its internals dropped. `b` must not be dropped.
    pub unsafe fn new<T: ?Sized>(b: &mut ErrorBox<T>) -> Self {
        Self {
            layout: Layout::for_value::<T>(&**b),
            // SAFETY: guaranteed safe by caller
            ptr: unsafe { NonNull::new_unchecked(b.0.as_ptr().cast()) },
        }
    }
}

impl Drop for ErrorBoxMem {
    fn drop(&mut self) {
        // SAFETY: guaranteed safe by `ErrorBoxMem::new`
        unsafe { (AllocVTable::get().deallocate)(self.ptr, self.layout) }
    }
}

impl<T: ?Sized> Drop for ErrorBox<T> {
    fn drop(&mut self) {
        // SAFETY: guaranteed safe by ErrorBox's safety invariant
        unsafe {
            let _mem = ErrorBoxMem::new(self);
            ptr::drop_in_place(self.0.as_ptr());
        }
    }
}

/// implementation detail of [`Error`]
#[derive(Debug)]
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub struct AllocVTable {
    /// implementation detail of [`Error`]
    ///
    /// Safety: same as [`Allocator::deallocate`]
    ///
    /// [`Allocator::deallocate`]: core::alloc::Allocator::deallocate
    pub deallocate: unsafe fn(ptr: NonNull<u8>, layout: Layout),
}

static ALLOC_VTABLE: AtomicPtr<AllocVTable> = AtomicPtr::new(ptr::null_mut());

impl AllocVTable {
    /// implementation detail of [`Error`]
    ///
    /// Safety: `self` must be in a `static` variable that has never been
    /// written. All members must be valid.
    pub unsafe fn install(&'static self) {
        // see `get` for why `Relaxed` is sufficient.
        if ALLOC_VTABLE.load(Ordering::Relaxed).is_null() {
            ALLOC_VTABLE.store(<*const _>::from(self).cast_mut(), Ordering::Relaxed);
        }
    }
    /// implementation detail of [`Error`]
    ///
    /// Safety: `install` must have been called
    pub unsafe fn get() -> &'static Self {
        // SAFETY: `install` has been called and it set `ALLOC_VTABLE` to point
        // to a `static` variable that has never been written, so no writes can
        // possibly need to be communicated through other memory so `Relaxed`
        // here and in `install` are sufficient.
        unsafe { &*ALLOC_VTABLE.load(Ordering::Relaxed) }
    }
}

/// implementation detail of [`Error`]
#[derive(Debug)]
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub struct StdVTable {
    /// implementation detail of [`Error`]
    pub decode_error_kind: fn(RawOsError) -> ErrorKind,
    /// implementation detail of [`Error`]
    pub error_string: fn(RawOsError) -> ErrorString,
}

static STD_VTABLE: AtomicPtr<StdVTable> = AtomicPtr::new(ptr::null_mut());

impl StdVTable {
    /// implementation detail of [`Error`]
    ///
    /// Safety: `self` must be in a `static` variable that has never been
    /// written. All members must be valid.
    pub unsafe fn install(&'static self) {
        // see `get` for why `Relaxed` is sufficient.
        if STD_VTABLE.load(Ordering::Relaxed).is_null() {
            STD_VTABLE.store(<*const _>::from(self).cast_mut(), Ordering::Relaxed);
        }
    }
    /// implementation detail of [`Error`]
    ///
    /// Safety: `install` must have been called
    pub unsafe fn get() -> &'static Self {
        // SAFETY: `install` has been called and it set `STD_VTABLE` to point
        // to a `static` variable that has never been written, so no writes can
        // possibly need to be communicated through other memory so `Relaxed`
        // here and in `install` are sufficient.
        unsafe { &*STD_VTABLE.load(Ordering::Relaxed) }
    }
}

/// A specialized [`Result`] type for I/O operations.
///
/// This type is broadly used across [`std::io`] for any operation which may
/// produce an error.
///
/// This typedef is generally used to avoid writing out [`io::Error`] directly and
/// is otherwise a direct mapping to [`Result`].
///
/// While usual Rust style is to import types directly, aliases of [`Result`]
/// often are not, to make it easier to distinguish between them. [`Result`] is
/// generally assumed to be [`std::result::Result`][`Result`], and so users of this alias
/// will generally use `io::Result` instead of shadowing the [prelude]'s import
/// of [`std::result::Result`][`Result`].
///
/// [`std::io`]: https://doc.rust-lang.org/std/io/index.html
/// [`io::Error`]: https://doc.rust-lang.org/std/io/struct.Error.html
/// [`Result`]: https://doc.rust-lang.org/std/result/enum.Result.html
/// [prelude]: https://doc.rust-lang.org/std/prelude/index.html
///
/// # Examples
///
/// A convenience function that bubbles an `io::Result` to its caller:
///
/// ```
/// use std::io;
///
/// fn get_string() -> io::Result<String> {
///     let mut buffer = String::new();
///
///     io::stdin().read_line(&mut buffer)?;
///
///     Ok(buffer)
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub type Result<T> = result::Result<T, Error>;

/// The error type for I/O operations of the [`Read`], [`Write`], [`Seek`], and
/// associated traits.
///
/// Errors mostly originate from the underlying OS, but custom instances of
/// `Error` can be created with crafted error messages and a particular value of
/// [`ErrorKind`].
///
/// [`Read`]: https://doc.rust-lang.org/std/io/trait.Read.html
/// [`Write`]: https://doc.rust-lang.org/std/io/trait.Write.html
/// [`Seek`]: https://doc.rust-lang.org/std/io/trait.Seek.html
// Safety: [`StdVTable::install`] must have been run before any `Error`
// instances containing OS errors are created.
// [`AllocVTable::install`] must have been run before any instances containing
// `AllocBox` are created.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_has_incoherent_inherent_impls]
pub struct Error {
    repr: Repr,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.repr, f)
    }
}

// Only derive debug in tests, to make sure it
// doesn't accidentally get printed.
#[cfg_attr(test, derive(Debug))]
enum ErrorData<C> {
    Os(RawOsError),
    Simple(ErrorKind),
    SimpleMessage(&'static SimpleMessage),
    Custom(C),
}

#[cfg(not(target_os = "uefi"))]
type RawOsError_ = i32;

#[cfg(target_os = "uefi")]
type RawOsError_ = usize;

/// The type of raw OS error codes returned by [`Error::raw_os_error`].
///
/// This is an [`i32`] on all currently supported platforms, but platforms
/// added in the future (such as UEFI) may use a different primitive type like
/// [`usize`]. Use `as`or [`into`] conversions where applicable to ensure maximum
/// portability.
///
/// [`into`]: Into::into
#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub type RawOsError = RawOsError_;

// `#[repr(align(4))]` is probably redundant, it should have that value or
// higher already. We include it just because repr_bitpacked.rs's encoding
// requires an alignment >= 4 (note that `#[repr(align)]` will not reduce the
// alignment required by the struct, only increase it).
//
// If we add more variants to ErrorData, this can be increased to 8, but it
// should probably be behind `#[cfg_attr(target_pointer_width = "64", ...)]` or
// whatever cfg we're using to enable the `repr_bitpacked` code, since only the
// that version needs the alignment, and 8 is higher than the alignment we'll
// have on 32 bit platforms.
//
// (For the sake of being explicit: the alignment requirement here only matters
// if `error/repr_bitpacked.rs` is in use — for the unpacked repr it doesn't
// matter at all)
#[unstable(feature = "core_io_error_internals", issue = "none")]
#[repr(align(4))]
#[derive(Debug)]
/// implementation detail of [`Error`]
pub struct SimpleMessage {
    kind: ErrorKind,
    message: &'static str,
}

impl SimpleMessage {
    /// implementation detail of [`Error`]
    #[unstable(feature = "core_io_error_internals", issue = "none")]
    pub const fn new(kind: ErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }
}

/// Create and return an `io::Error` for a given `ErrorKind` and constant
/// message. This doesn't allocate.
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub macro const_io_error($kind:expr, $message:expr $(,)?) {
    $crate::io::error::Error::from_static_message({
        const MESSAGE_DATA: $crate::io::error::SimpleMessage =
            $crate::io::error::SimpleMessage::new($kind, $message);
        &MESSAGE_DATA
    })
}

// As with `SimpleMessage`: `#[repr(align(4))]` here is just because
// repr_bitpacked's encoding requires it. In practice it almost certainly be
// already be this high or higher.
#[derive(Debug)]
#[repr(align(4))]
#[unstable(feature = "core_io_error_internals", issue = "none")]
/// implementation detail of [`Error`]
pub struct Custom {
    /// implementation detail of [`Error`]
    pub kind: ErrorKind,
    /// implementation detail of [`Error`]
    pub error: ErrorBox<dyn error::Error + Send + Sync>,
}

/// A list specifying general categories of I/O error.
///
/// This list is intended to grow over time and it is not recommended to
/// exhaustively match against it.
///
/// It is used with the [`io::Error`] type.
///
/// [`io::Error`]: Error
///
/// # Handling errors and matching on `ErrorKind`
///
/// In application code, use `match` for the `ErrorKind` values you are
/// expecting; use `_` to match "all other errors".
///
/// In comprehensive and thorough tests that want to verify that a test doesn't
/// return any known incorrect error kind, you may want to cut-and-paste the
/// current full list of errors from here into your test code, and then match
/// `_` as the correct case. This seems counterintuitive, but it will make your
/// tests more robust. In particular, if you want to verify that your code does
/// produce an unrecognized error kind, the robust solution is to check for all
/// the recognized error kinds and fail in those cases.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An entity was not found, often a file.
    #[stable(feature = "rust1", since = "1.0.0")]
    NotFound,
    /// The operation lacked the necessary privileges to complete.
    #[stable(feature = "rust1", since = "1.0.0")]
    PermissionDenied,
    /// The connection was refused by the remote server.
    #[stable(feature = "rust1", since = "1.0.0")]
    ConnectionRefused,
    /// The connection was reset by the remote server.
    #[stable(feature = "rust1", since = "1.0.0")]
    ConnectionReset,
    /// The remote host is not reachable.
    #[unstable(feature = "io_error_more", issue = "86442")]
    HostUnreachable,
    /// The network containing the remote host is not reachable.
    #[unstable(feature = "io_error_more", issue = "86442")]
    NetworkUnreachable,
    /// The connection was aborted (terminated) by the remote server.
    #[stable(feature = "rust1", since = "1.0.0")]
    ConnectionAborted,
    /// The network operation failed because it was not connected yet.
    #[stable(feature = "rust1", since = "1.0.0")]
    NotConnected,
    /// A socket address could not be bound because the address is already in
    /// use elsewhere.
    #[stable(feature = "rust1", since = "1.0.0")]
    AddrInUse,
    /// A nonexistent interface was requested or the requested address was not
    /// local.
    #[stable(feature = "rust1", since = "1.0.0")]
    AddrNotAvailable,
    /// The system's networking is down.
    #[unstable(feature = "io_error_more", issue = "86442")]
    NetworkDown,
    /// The operation failed because a pipe was closed.
    #[stable(feature = "rust1", since = "1.0.0")]
    BrokenPipe,
    /// An entity already exists, often a file.
    #[stable(feature = "rust1", since = "1.0.0")]
    AlreadyExists,
    /// The operation needs to block to complete, but the blocking operation was
    /// requested to not occur.
    #[stable(feature = "rust1", since = "1.0.0")]
    WouldBlock,
    /// A filesystem object is, unexpectedly, not a directory.
    ///
    /// For example, a filesystem path was specified where one of the intermediate directory
    /// components was, in fact, a plain file.
    #[unstable(feature = "io_error_more", issue = "86442")]
    NotADirectory,
    /// The filesystem object is, unexpectedly, a directory.
    ///
    /// A directory was specified when a non-directory was expected.
    #[unstable(feature = "io_error_more", issue = "86442")]
    IsADirectory,
    /// A non-empty directory was specified where an empty directory was expected.
    #[unstable(feature = "io_error_more", issue = "86442")]
    DirectoryNotEmpty,
    /// The filesystem or storage medium is read-only, but a write operation was attempted.
    #[unstable(feature = "io_error_more", issue = "86442")]
    ReadOnlyFilesystem,
    /// Loop in the filesystem or IO subsystem; often, too many levels of symbolic links.
    ///
    /// There was a loop (or excessively long chain) resolving a filesystem object
    /// or file IO object.
    ///
    /// On Unix this is usually the result of a symbolic link loop; or, of exceeding the
    /// system-specific limit on the depth of symlink traversal.
    #[unstable(feature = "io_error_more", issue = "86442")]
    FilesystemLoop,
    /// Stale network file handle.
    ///
    /// With some network filesystems, notably NFS, an open file (or directory) can be invalidated
    /// by problems with the network or server.
    #[unstable(feature = "io_error_more", issue = "86442")]
    StaleNetworkFileHandle,
    /// A parameter was incorrect.
    #[stable(feature = "rust1", since = "1.0.0")]
    InvalidInput,
    /// Data not valid for the operation were encountered.
    ///
    /// Unlike [`InvalidInput`], this typically means that the operation
    /// parameters were valid, however the error was caused by malformed
    /// input data.
    ///
    /// For example, a function that reads a file into a string will error with
    /// `InvalidData` if the file's contents are not valid UTF-8.
    ///
    /// [`InvalidInput`]: ErrorKind::InvalidInput
    #[stable(feature = "io_invalid_data", since = "1.2.0")]
    InvalidData,
    /// The I/O operation's timeout expired, causing it to be canceled.
    #[stable(feature = "rust1", since = "1.0.0")]
    TimedOut,
    /// An error returned when an operation could not be completed because a
    /// call to [`write`] returned [`Ok(0)`].
    ///
    /// This typically means that an operation could only succeed if it wrote a
    /// particular number of bytes but only a smaller number of bytes could be
    /// written.
    ///
    /// [`write`]: https://doc.rust-lang.org/std/io/trait.Write.html#tymethod.write
    /// [`Ok(0)`]: Ok
    #[stable(feature = "rust1", since = "1.0.0")]
    WriteZero,
    /// The underlying storage (typically, a filesystem) is full.
    ///
    /// This does not include out of quota errors.
    #[unstable(feature = "io_error_more", issue = "86442")]
    StorageFull,
    /// Seek on unseekable file.
    ///
    /// Seeking was attempted on an open file handle which is not suitable for seeking - for
    /// example, on Unix, a named pipe opened with `File::open`.
    #[unstable(feature = "io_error_more", issue = "86442")]
    NotSeekable,
    /// Filesystem quota was exceeded.
    #[unstable(feature = "io_error_more", issue = "86442")]
    FilesystemQuotaExceeded,
    /// File larger than allowed or supported.
    ///
    /// This might arise from a hard limit of the underlying filesystem or file access API, or from
    /// an administratively imposed resource limitation.  Simple disk full, and out of quota, have
    /// their own errors.
    #[unstable(feature = "io_error_more", issue = "86442")]
    FileTooLarge,
    /// Resource is busy.
    #[unstable(feature = "io_error_more", issue = "86442")]
    ResourceBusy,
    /// Executable file is busy.
    ///
    /// An attempt was made to write to a file which is also in use as a running program.  (Not all
    /// operating systems detect this situation.)
    #[unstable(feature = "io_error_more", issue = "86442")]
    ExecutableFileBusy,
    /// Deadlock (avoided).
    ///
    /// A file locking operation would result in deadlock.  This situation is typically detected, if
    /// at all, on a best-effort basis.
    #[unstable(feature = "io_error_more", issue = "86442")]
    Deadlock,
    /// Cross-device or cross-filesystem (hard) link or rename.
    #[unstable(feature = "io_error_more", issue = "86442")]
    CrossesDevices,
    /// Too many (hard) links to the same filesystem object.
    ///
    /// The filesystem does not support making so many hardlinks to the same file.
    #[unstable(feature = "io_error_more", issue = "86442")]
    TooManyLinks,
    /// A filename was invalid.
    ///
    /// This error can also cause if it exceeded the filename length limit.
    #[unstable(feature = "io_error_more", issue = "86442")]
    InvalidFilename,
    /// Program argument list too long.
    ///
    /// When trying to run an external program, a system or process limit on the size of the
    /// arguments would have been exceeded.
    #[unstable(feature = "io_error_more", issue = "86442")]
    ArgumentListTooLong,
    /// This operation was interrupted.
    ///
    /// Interrupted operations can typically be retried.
    #[stable(feature = "rust1", since = "1.0.0")]
    Interrupted,

    /// This operation is unsupported on this platform.
    ///
    /// This means that the operation can never succeed.
    #[stable(feature = "unsupported_error", since = "1.53.0")]
    Unsupported,

    // ErrorKinds which are primarily categorisations for OS error
    // codes should be added above.
    //
    /// An error returned when an operation could not be completed because an
    /// "end of file" was reached prematurely.
    ///
    /// This typically means that an operation could only succeed if it read a
    /// particular number of bytes but only a smaller number of bytes could be
    /// read.
    #[stable(feature = "read_exact", since = "1.6.0")]
    UnexpectedEof,

    /// An operation could not be completed, because it failed
    /// to allocate enough memory.
    #[stable(feature = "out_of_memory_error", since = "1.54.0")]
    OutOfMemory,

    // "Unusual" error kinds which do not correspond simply to (sets
    // of) OS error codes, should be added just above this comment.
    // `Other` and `Uncategorized` should remain at the end:
    //
    /// A custom error that does not fall under any other I/O error kind.
    ///
    /// This can be used to construct your own [`Error`]s that do not match any
    /// [`ErrorKind`].
    ///
    /// This [`ErrorKind`] is not used by the standard library.
    ///
    /// Errors from the standard library that do not fall under any of the I/O
    /// error kinds cannot be `match`ed on, and will only match a wildcard (`_`) pattern.
    /// New [`ErrorKind`]s might be added in the future for some of those.
    #[stable(feature = "rust1", since = "1.0.0")]
    Other,

    /// Any I/O error from the standard library that's not part of this list.
    ///
    /// Errors that are `Uncategorized` now may move to a different or a new
    /// [`ErrorKind`] variant in the future. It is not recommended to match
    /// an error against `Uncategorized`; use a wildcard match (`_`) instead.
    #[unstable(feature = "io_error_uncategorized", issue = "none")]
    #[doc(hidden)]
    Uncategorized,
}

impl ErrorKind {
    /// implementation detail of [`Error`]
    #[unstable(feature = "core_io_error_internals", issue = "none")]
    pub fn as_str(&self) -> &'static str {
        use ErrorKind::*;
        // tidy-alphabetical-start
        match *self {
            AddrInUse => "address in use",
            AddrNotAvailable => "address not available",
            AlreadyExists => "entity already exists",
            ArgumentListTooLong => "argument list too long",
            BrokenPipe => "broken pipe",
            ConnectionAborted => "connection aborted",
            ConnectionRefused => "connection refused",
            ConnectionReset => "connection reset",
            CrossesDevices => "cross-device link or rename",
            Deadlock => "deadlock",
            DirectoryNotEmpty => "directory not empty",
            ExecutableFileBusy => "executable file busy",
            FileTooLarge => "file too large",
            FilesystemLoop => "filesystem loop or indirection limit (e.g. symlink loop)",
            FilesystemQuotaExceeded => "filesystem quota exceeded",
            HostUnreachable => "host unreachable",
            Interrupted => "operation interrupted",
            InvalidData => "invalid data",
            InvalidFilename => "invalid filename",
            InvalidInput => "invalid input parameter",
            IsADirectory => "is a directory",
            NetworkDown => "network down",
            NetworkUnreachable => "network unreachable",
            NotADirectory => "not a directory",
            NotConnected => "not connected",
            NotFound => "entity not found",
            NotSeekable => "seek on unseekable file",
            Other => "other error",
            OutOfMemory => "out of memory",
            PermissionDenied => "permission denied",
            ReadOnlyFilesystem => "read-only filesystem or storage medium",
            ResourceBusy => "resource busy",
            StaleNetworkFileHandle => "stale network file handle",
            StorageFull => "no storage space",
            TimedOut => "timed out",
            TooManyLinks => "too many links",
            Uncategorized => "uncategorized error",
            UnexpectedEof => "unexpected end of file",
            Unsupported => "unsupported",
            WouldBlock => "operation would block",
            WriteZero => "write zero",
        }
        // tidy-alphabetical-end
    }
}

#[stable(feature = "io_errorkind_display", since = "1.60.0")]
impl fmt::Display for ErrorKind {
    /// Shows a human-readable description of the `ErrorKind`.
    ///
    /// This is similar to `impl Display for Error`, but doesn't require first converting to Error.
    ///
    /// # Examples
    /// ```
    /// use std::io::ErrorKind;
    /// assert_eq!("entity not found", ErrorKind::NotFound.to_string());
    /// ```
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.as_str())
    }
}

/// Intended for use for errors not exposed to the user, where allocating onto
/// the heap (for normal construction via Error::new) is too costly.
#[stable(feature = "io_error_from_errorkind", since = "1.14.0")]
impl From<ErrorKind> for Error {
    /// Converts an [`ErrorKind`] into an [`Error`].
    ///
    /// This conversion creates a new error with a simple representation of error kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// let not_found = ErrorKind::NotFound;
    /// let error = Error::from(not_found);
    /// assert_eq!("entity not found", format!("{error}"));
    /// ```
    #[inline]
    fn from(kind: ErrorKind) -> Error {
        Error { repr: Repr::new_simple(kind) }
    }
}

impl Error {
    /// implementation detail of [`Error::new`]
    ///
    /// [`Error::new`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.new
    #[unstable(feature = "core_io_error_internals", issue = "none")]
    pub fn _new_custom(v: ErrorBox<Custom>) -> Error {
        Error { repr: Repr::new_custom(v) }
    }

    /// Creates a new I/O error from a known kind of error as well as a constant
    /// message.
    ///
    /// This function does not allocate.
    ///
    /// You should not use this directly, and instead use the `const_io_error!`
    /// macro: `io::const_io_error!(ErrorKind::Something, "some_message")`.
    ///
    /// This function should maybe change to `from_static_message<const MSG: &'static
    /// str>(kind: ErrorKind)` in the future, when const generics allow that.
    #[inline]
    #[unstable(feature = "core_io_error_internals", issue = "none")]
    pub const fn from_static_message(msg: &'static SimpleMessage) -> Error {
        Self { repr: Repr::new_simple_message(msg) }
    }

    /// implementation detail of [`Error::from_raw_os_error`]
    ///
    /// [`Error::from_raw_os_error`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.from_raw_os_error
    #[unstable(feature = "core_io_error_internals", issue = "none")]
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_os_error_impl(code: RawOsError) -> Error {
        Error { repr: Repr::new_os(code) }
    }

    /// Returns the OS error that this error represents (if any).
    ///
    /// If this [`Error`] was constructed via [`last_os_error`] or
    /// [`from_raw_os_error`], then this function will return [`Some`], otherwise
    /// it will return [`None`].
    ///
    /// [`last_os_error`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.last_os_error
    /// [`from_raw_os_error`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.from_raw_os_error
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_os_error(err: &Error) {
    ///     if let Some(raw_os_err) = err.raw_os_error() {
    ///         println!("raw OS error: {raw_os_err:?}");
    ///     } else {
    ///         println!("Not an OS error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "raw OS error: ...".
    ///     print_os_error(&Error::last_os_error());
    ///     // Will print "Not an OS error".
    ///     print_os_error(&Error::new(ErrorKind::Other, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn raw_os_error(&self) -> Option<RawOsError> {
        match self.repr.data() {
            ErrorData::Os(i) => Some(i),
            ErrorData::Custom(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
        }
    }

    /// Returns a reference to the inner error wrapped by this error (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [`new`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_error(err: &Error) {
    ///     if let Some(inner_err) = err.get_ref() {
    ///         println!("Inner error: {inner_err:?}");
    ///     } else {
    ///         println!("No inner error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "No inner error".
    ///     print_error(&Error::last_os_error());
    ///     // Will print "Inner error: ...".
    ///     print_error(&Error::new(ErrorKind::Other, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "io_error_inner", since = "1.3.0")]
    #[must_use]
    #[inline]
    pub fn get_ref(&self) -> Option<&(dyn error::Error + Send + Sync + 'static)> {
        match self.repr.data() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(&*c.error),
        }
    }

    /// Returns a mutable reference to the inner error wrapped by this error
    /// (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [`new`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    /// use std::{error, fmt};
    /// use std::fmt::Display;
    ///
    /// #[derive(Debug)]
    /// struct MyError {
    ///     v: String,
    /// }
    ///
    /// impl MyError {
    ///     fn new() -> MyError {
    ///         MyError {
    ///             v: "oh no!".to_string()
    ///         }
    ///     }
    ///
    ///     fn change_message(&mut self, new_message: &str) {
    ///         self.v = new_message.to_string();
    ///     }
    /// }
    ///
    /// impl error::Error for MyError {}
    ///
    /// impl Display for MyError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "MyError: {}", &self.v)
    ///     }
    /// }
    ///
    /// fn change_error(mut err: Error) -> Error {
    ///     if let Some(inner_err) = err.get_mut() {
    ///         inner_err.downcast_mut::<MyError>().unwrap().change_message("I've been changed!");
    ///     }
    ///     err
    /// }
    ///
    /// fn print_error(err: &Error) {
    ///     if let Some(inner_err) = err.get_ref() {
    ///         println!("Inner error: {inner_err}");
    ///     } else {
    ///         println!("No inner error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "No inner error".
    ///     print_error(&change_error(Error::last_os_error()));
    ///     // Will print "Inner error: ...".
    ///     print_error(&change_error(Error::new(ErrorKind::Other, MyError::new())));
    /// }
    /// ```
    #[stable(feature = "io_error_inner", since = "1.3.0")]
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut (dyn error::Error + Send + Sync + 'static)> {
        match self.repr.data_mut() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(&mut *c.error),
        }
    }

    /// implementation detail of [`Error::into_inner`]
    ///
    /// [`Error::into_inner`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.into_inner
    #[unstable(feature = "core_io_error_internals", issue = "none")]
    #[inline]
    pub fn into_inner_impl(self) -> Option<ErrorBox<dyn error::Error + Send + Sync>> {
        match self.repr.into_data() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(c.into_inner().error),
        }
    }

    /// implementation detail of [`Error::downcast`]
    ///
    /// [`Error::downcast`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.downcast
    #[unstable(feature = "core_io_error_internals", issue = "none")]
    pub fn downcast_impl<E>(self) -> result::Result<*mut E, Self>
    where
        E: error::Error + Send + Sync + 'static,
    {
        match self.repr.into_data() {
            ErrorData::Custom(b) if b.error.is::<E>() => Ok(b.into_inner().error.into_raw().cast()),
            repr_data => Err(Self { repr: Repr::new(repr_data) }),
        }
    }

    /// Returns the corresponding [`ErrorKind`] for this error.
    ///
    /// This may be a value set by Rust code constructing custom `io::Error`s,
    /// or if this `io::Error` was sourced from the operating system,
    /// it will be a value inferred from the system's error encoding.
    /// See [`last_os_error`] for more details.
    ///
    /// [`last_os_error`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.last_os_error
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_error(err: Error) {
    ///     println!("{:?}", err.kind());
    /// }
    ///
    /// fn main() {
    ///     // As no error has (visibly) occurred, this may print anything!
    ///     // It likely prints a placeholder for unidentified (non-)errors.
    ///     print_error(Error::last_os_error());
    ///     // Will print "AddrInUse".
    ///     print_error(Error::new(ErrorKind::AddrInUse, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        match self.repr.data() {
            // SAFETY: `Error`'s safety invariant guarantees
            // `StdVTable::install` has been run
            ErrorData::Os(code) => unsafe { (StdVTable::get().decode_error_kind)(code) },
            ErrorData::Custom(c) => c.kind,
            ErrorData::Simple(kind) => kind,
            ErrorData::SimpleMessage(m) => m.kind,
        }
    }
}

impl fmt::Debug for Repr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data() {
            ErrorData::Os(code) => {
                // SAFETY: `Error`'s safety invariant guarantees
                // `StdVTable::install` has been run
                let std_vtable = unsafe { StdVTable::get() };
                fmt.debug_struct("Os")
                    .field("code", &code)
                    .field("kind", &(std_vtable.decode_error_kind)(code))
                    .field("message", &(std_vtable.error_string)(code))
                    .finish()
            }
            ErrorData::Custom(c) => fmt::Debug::fmt(&c, fmt),
            ErrorData::Simple(kind) => fmt.debug_tuple("Kind").field(&kind).finish(),
            ErrorData::SimpleMessage(msg) => fmt
                .debug_struct("Error")
                .field("kind", &msg.kind)
                .field("message", &msg.message)
                .finish(),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.repr.data() {
            ErrorData::Os(code) => {
                // SAFETY: `Error`'s safety invariant guarantees
                // `StdVTable::install` has been run
                let detail = unsafe { (StdVTable::get().error_string)(code) };
                write!(fmt, "{detail} (os error {code})")
            }
            ErrorData::Custom(ref c) => c.error.fmt(fmt),
            ErrorData::Simple(kind) => write!(fmt, "{}", kind.as_str()),
            ErrorData::SimpleMessage(msg) => msg.message.fmt(fmt),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for Error {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        match self.repr.data() {
            ErrorData::Os(..) | ErrorData::Simple(..) => self.kind().as_str(),
            ErrorData::SimpleMessage(msg) => msg.message,
            ErrorData::Custom(c) => c.error.description(),
        }
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn error::Error> {
        match self.repr.data() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error.cause(),
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self.repr.data() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error.source(),
        }
    }
}

fn _assert_error_is_sync_send() {
    fn _is_sync_send<T: Sync + Send>() {}
    _is_sync_send::<Error>();
}
