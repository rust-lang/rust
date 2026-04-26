#![unstable(feature = "core_io", issue = "154046")]

use crate::fmt;

/// A list specifying general categories of I/O error.
///
/// This list is intended to grow over time and it is not recommended to
/// exhaustively match against it.
///
/// It is used with the [`io::Error`][error] type.
///
/// [error]: ../../std/io/struct.Error.html
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
#[cfg_attr(not(test), rustc_diagnostic_item = "io_errorkind")]
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
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    HostUnreachable,
    /// The network containing the remote host is not reachable.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
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
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
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
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    NotADirectory,
    /// The filesystem object is, unexpectedly, a directory.
    ///
    /// A directory was specified when a non-directory was expected.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    IsADirectory,
    /// A non-empty directory was specified where an empty directory was expected.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    DirectoryNotEmpty,
    /// The filesystem or storage medium is read-only, but a write operation was attempted.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
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
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
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
    /// call to [`write`][write] returned [`Ok(0)`].
    ///
    /// This typically means that an operation could only succeed if it wrote a
    /// particular number of bytes but only a smaller number of bytes could be
    /// written.
    ///
    /// [write]: ../../std/io/trait.Write.html#tymethod.write
    /// [`Ok(0)`]: Ok
    #[stable(feature = "rust1", since = "1.0.0")]
    WriteZero,
    /// The underlying storage (typically, a filesystem) is full.
    ///
    /// This does not include out of quota errors.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    StorageFull,
    /// Seek on unseekable file.
    ///
    /// Seeking was attempted on an open file handle which is not suitable for seeking - for
    /// example, on Unix, a named pipe opened with `File::open`.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    NotSeekable,
    /// Filesystem quota or some other kind of quota was exceeded.
    #[stable(feature = "io_error_quota_exceeded", since = "1.85.0")]
    QuotaExceeded,
    /// File larger than allowed or supported.
    ///
    /// This might arise from a hard limit of the underlying filesystem or file access API, or from
    /// an administratively imposed resource limitation.  Simple disk full, and out of quota, have
    /// their own errors.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    FileTooLarge,
    /// Resource is busy.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    ResourceBusy,
    /// Executable file is busy.
    ///
    /// An attempt was made to write to a file which is also in use as a running program.  (Not all
    /// operating systems detect this situation.)
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    ExecutableFileBusy,
    /// Deadlock (avoided).
    ///
    /// A file locking operation would result in deadlock.  This situation is typically detected, if
    /// at all, on a best-effort basis.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    Deadlock,
    /// Cross-device or cross-filesystem (hard) link or rename.
    #[stable(feature = "io_error_crosses_devices", since = "1.85.0")]
    CrossesDevices,
    /// Too many (hard) links to the same filesystem object.
    ///
    /// The filesystem does not support making so many hardlinks to the same file.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
    TooManyLinks,
    /// A filename was invalid.
    ///
    /// This error can also occur if a length limit for a name was exceeded.
    #[stable(feature = "io_error_invalid_filename", since = "1.87.0")]
    InvalidFilename,
    /// Program argument list too long.
    ///
    /// When trying to run an external program, a system or process limit on the size of the
    /// arguments would have been exceeded.
    #[stable(feature = "io_error_a_bit_more", since = "1.83.0")]
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

    /// The operation was partially successful and needs to be checked
    /// later on due to not blocking.
    #[unstable(feature = "io_error_inprogress", issue = "130840")]
    InProgress,

    // "Unusual" error kinds which do not correspond simply to (sets
    // of) OS error codes, should be added just above this comment.
    // `Other` and `Uncategorized` should remain at the end:
    //
    /// A custom error that does not fall under any other I/O error kind.
    ///
    /// This can be used to construct your own [`Error`][error]s that do not match any
    /// [`ErrorKind`].
    ///
    /// This [`ErrorKind`] is not used by the standard library.
    ///
    /// Errors from the standard library that do not fall under any of the I/O
    /// error kinds cannot be `match`ed on, and will only match a wildcard (`_`) pattern.
    /// New [`ErrorKind`]s might be added in the future for some of those.
    ///
    /// [error]: ../../std/io/struct.Error.html
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
    pub(crate) const fn as_str(&self) -> &'static str {
        use ErrorKind::*;
        match *self {
            // tidy-alphabetical-start
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
            HostUnreachable => "host unreachable",
            InProgress => "in progress",
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
            QuotaExceeded => "quota exceeded",
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
            // tidy-alphabetical-end
        }
    }

    // This compiles to the same code as the check+transmute, but doesn't require
    // unsafe, or to hard-code max ErrorKind or its size in a way the compiler
    // couldn't verify.
    #[inline]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    #[doc(hidden)]
    pub const fn from_prim(ek: u32) -> Option<Self> {
        macro_rules! from_prim {
            ($prim:expr => $Enum:ident { $($Variant:ident),* $(,)? }) => {{
                // Force a compile error if the list gets out of date.
                const _: fn(e: $Enum) = |e: $Enum| match e {
                    $($Enum::$Variant => (),)*
                };
                match $prim {
                    $(v if v == ($Enum::$Variant as _) => Some($Enum::$Variant),)*
                    _ => None,
                }
            }}
        }
        from_prim!(ek => ErrorKind {
            NotFound,
            PermissionDenied,
            ConnectionRefused,
            ConnectionReset,
            HostUnreachable,
            NetworkUnreachable,
            ConnectionAborted,
            NotConnected,
            AddrInUse,
            AddrNotAvailable,
            NetworkDown,
            BrokenPipe,
            AlreadyExists,
            WouldBlock,
            NotADirectory,
            IsADirectory,
            DirectoryNotEmpty,
            ReadOnlyFilesystem,
            FilesystemLoop,
            StaleNetworkFileHandle,
            InvalidInput,
            InvalidData,
            TimedOut,
            WriteZero,
            StorageFull,
            NotSeekable,
            QuotaExceeded,
            FileTooLarge,
            ResourceBusy,
            ExecutableFileBusy,
            Deadlock,
            CrossesDevices,
            TooManyLinks,
            InvalidFilename,
            ArgumentListTooLong,
            Interrupted,
            Other,
            UnexpectedEof,
            Unsupported,
            OutOfMemory,
            InProgress,
            Uncategorized,
        })
    }
}

#[stable(feature = "io_errorkind_display", since = "1.60.0")]
impl fmt::Display for ErrorKind {
    /// Shows a human-readable description of the [`ErrorKind`].
    ///
    /// This is similar to `impl Display for Error`, but doesn't require first converting to Error.
    ///
    /// # Examples
    /// ```
    /// use core::io::ErrorKind;
    /// assert_eq!("entity not found", ErrorKind::NotFound.to_string());
    /// ```
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.as_str())
    }
}
