use std::io;
use std::io::ErrorKind;

use crate::*;

/// A representation of an IO error: either a libc error name,
/// or a host error.
#[derive(Debug)]
pub enum IoError {
    LibcError(&'static str),
    WindowsError(&'static str),
    HostError(io::Error),
    Raw(Scalar),
}
pub use self::IoError::*;

impl IoError {
    pub(crate) fn into_ntstatus(self) -> i32 {
        let raw = match self {
            HostError(e) =>
                match e.kind() {
                    // STATUS_MEDIA_WRITE_PROTECTED
                    ErrorKind::ReadOnlyFilesystem => 0xC00000A2u32,
                    // STATUS_FILE_INVALID
                    ErrorKind::InvalidInput => 0xC0000098,
                    // STATUS_DISK_FULL
                    ErrorKind::QuotaExceeded => 0xC000007F,
                    // STATUS_ACCESS_DENIED
                    ErrorKind::PermissionDenied => 0xC0000022,
                    // For the default error code we arbitrarily pick 0xC0000185, STATUS_IO_DEVICE_ERROR.
                    _ => 0xC0000185,
                },
            // For the default error code we arbitrarily pick 0xC0000185, STATUS_IO_DEVICE_ERROR.
            _ => 0xC0000185,
        };
        raw.cast_signed()
    }
}

impl From<io::Error> for IoError {
    fn from(value: io::Error) -> Self {
        IoError::HostError(value)
    }
}

impl From<io::ErrorKind> for IoError {
    fn from(value: io::ErrorKind) -> Self {
        IoError::HostError(value.into())
    }
}

impl From<Scalar> for IoError {
    fn from(value: Scalar) -> Self {
        IoError::Raw(value)
    }
}

// This mapping should match `decode_error_kind` in
// <https://github.com/rust-lang/rust/blob/HEAD/library/std/src/sys/io/error/unix.rs>.
const UNIX_IO_ERROR_TABLE: &[(&str, std::io::ErrorKind)] = {
    use std::io::ErrorKind::*;
    &[
        ("E2BIG", ArgumentListTooLong),
        ("EADDRINUSE", AddrInUse),
        ("EADDRNOTAVAIL", AddrNotAvailable),
        ("EBUSY", ResourceBusy),
        ("ECONNABORTED", ConnectionAborted),
        ("ECONNREFUSED", ConnectionRefused),
        ("ECONNRESET", ConnectionReset),
        ("EDEADLK", Deadlock),
        ("EDQUOT", QuotaExceeded),
        ("EEXIST", AlreadyExists),
        ("EFBIG", FileTooLarge),
        ("EHOSTUNREACH", HostUnreachable),
        ("EINTR", Interrupted),
        ("EINVAL", InvalidInput),
        ("EISDIR", IsADirectory),
        ("ELOOP", FilesystemLoop),
        ("ENOENT", NotFound),
        ("ENOMEM", OutOfMemory),
        ("ENOSPC", StorageFull),
        ("ENOSYS", Unsupported),
        ("EMLINK", TooManyLinks),
        ("ENAMETOOLONG", InvalidFilename),
        ("ENETDOWN", NetworkDown),
        ("ENETUNREACH", NetworkUnreachable),
        ("ENOTCONN", NotConnected),
        ("ENOTDIR", NotADirectory),
        ("ENOTEMPTY", DirectoryNotEmpty),
        ("EPIPE", BrokenPipe),
        ("EROFS", ReadOnlyFilesystem),
        ("ESPIPE", NotSeekable),
        ("ESTALE", StaleNetworkFileHandle),
        ("ETIMEDOUT", TimedOut),
        ("ETXTBSY", ExecutableFileBusy),
        ("EXDEV", CrossesDevices),
        ("EINPROGRESS", InProgress),
        // The following have two valid options. We have both for the forwards mapping; only the
        // first one will be used for the backwards mapping.
        ("EPERM", PermissionDenied),
        ("EACCES", PermissionDenied),
        ("EWOULDBLOCK", WouldBlock),
        ("EAGAIN", WouldBlock),
    ]
};
// On Unix hosts are can avoid round-tripping via `ErrorKind`, which can preserve more
// details and leads to nicer output in `strerror_r`.
#[cfg(unix)]
const UNIX_ERRNO_TABLE: &[(&str, libc::c_int)] = &[
    ("E2BIG", libc::E2BIG),
    ("EACCES", libc::EACCES),
    ("EADDRINUSE", libc::EADDRINUSE),
    ("EADDRNOTAVAIL", libc::EADDRNOTAVAIL),
    ("EAFNOSUPPORT", libc::EAFNOSUPPORT),
    ("EAGAIN", libc::EAGAIN),
    ("EALREADY", libc::EALREADY),
    ("EBADF", libc::EBADF),
    ("EBADMSG", libc::EBADMSG),
    ("EBUSY", libc::EBUSY),
    ("ECANCELED", libc::ECANCELED),
    ("ECHILD", libc::ECHILD),
    ("ECONNABORTED", libc::ECONNABORTED),
    ("ECONNREFUSED", libc::ECONNREFUSED),
    ("ECONNRESET", libc::ECONNRESET),
    ("EDEADLK", libc::EDEADLK),
    ("EDESTADDRREQ", libc::EDESTADDRREQ),
    ("EDOM", libc::EDOM),
    ("EDQUOT", libc::EDQUOT),
    ("EEXIST", libc::EEXIST),
    ("EFAULT", libc::EFAULT),
    ("EFBIG", libc::EFBIG),
    ("EHOSTUNREACH", libc::EHOSTUNREACH),
    ("EIDRM", libc::EIDRM),
    ("EILSEQ", libc::EILSEQ),
    ("EINPROGRESS", libc::EINPROGRESS),
    ("EINTR", libc::EINTR),
    ("EINVAL", libc::EINVAL),
    ("EIO", libc::EIO),
    ("EISCONN", libc::EISCONN),
    ("EISDIR", libc::EISDIR),
    ("ELOOP", libc::ELOOP),
    ("EMFILE", libc::EMFILE),
    ("EMLINK", libc::EMLINK),
    ("EMSGSIZE", libc::EMSGSIZE),
    ("EMULTIHOP", libc::EMULTIHOP),
    ("ENAMETOOLONG", libc::ENAMETOOLONG),
    ("ENETDOWN", libc::ENETDOWN),
    ("ENETRESET", libc::ENETRESET),
    ("ENETUNREACH", libc::ENETUNREACH),
    ("ENFILE", libc::ENFILE),
    ("ENOBUFS", libc::ENOBUFS),
    ("ENODEV", libc::ENODEV),
    ("ENOENT", libc::ENOENT),
    ("ENOEXEC", libc::ENOEXEC),
    ("ENOLCK", libc::ENOLCK),
    ("ENOLINK", libc::ENOLINK),
    ("ENOMEM", libc::ENOMEM),
    ("ENOMSG", libc::ENOMSG),
    ("ENOPROTOOPT", libc::ENOPROTOOPT),
    ("ENOSPC", libc::ENOSPC),
    ("ENOSYS", libc::ENOSYS),
    ("ENOTCONN", libc::ENOTCONN),
    ("ENOTDIR", libc::ENOTDIR),
    ("ENOTEMPTY", libc::ENOTEMPTY),
    ("ENOTRECOVERABLE", libc::ENOTRECOVERABLE),
    ("ENOTSOCK", libc::ENOTSOCK),
    ("ENOTSUP", libc::ENOTSUP),
    ("ENOTTY", libc::ENOTTY),
    ("ENXIO", libc::ENXIO),
    ("EOPNOTSUPP", libc::EOPNOTSUPP),
    ("EOVERFLOW", libc::EOVERFLOW),
    ("EOWNERDEAD", libc::EOWNERDEAD),
    ("EPERM", libc::EPERM),
    ("EPIPE", libc::EPIPE),
    ("EPROTO", libc::EPROTO),
    ("EPROTONOSUPPORT", libc::EPROTONOSUPPORT),
    ("EPROTOTYPE", libc::EPROTOTYPE),
    ("ERANGE", libc::ERANGE),
    ("EROFS", libc::EROFS),
    ("ESOCKTNOSUPPORT", libc::ESOCKTNOSUPPORT),
    ("ESPIPE", libc::ESPIPE),
    ("ESRCH", libc::ESRCH),
    ("ESTALE", libc::ESTALE),
    ("ETIMEDOUT", libc::ETIMEDOUT),
    ("ETXTBSY", libc::ETXTBSY),
    ("EWOULDBLOCK", libc::EWOULDBLOCK),
    ("EXDEV", libc::EXDEV),
];
// This mapping should match `decode_error_kind` in
// <https://github.com/rust-lang/rust/blob/HEAD/library/std/src/sys/io/error/windows.rs>.
const WINDOWS_IO_ERROR_TABLE: &[(&str, std::io::ErrorKind)] = {
    use std::io::ErrorKind::*;
    // It's common for multiple error codes to map to the same io::ErrorKind. We have all for the
    // forwards mapping; only the first one will be used for the backwards mapping.
    // Slightly arbitrarily, we prefer non-WSA and the most generic sounding variant for backwards
    // mapping.
    &[
        ("WSAEADDRINUSE", AddrInUse),
        ("WSAEADDRNOTAVAIL", AddrNotAvailable),
        ("ERROR_ALREADY_EXISTS", AlreadyExists),
        ("ERROR_FILE_EXISTS", AlreadyExists),
        ("ERROR_NO_DATA", BrokenPipe),
        ("WSAECONNABORTED", ConnectionAborted),
        ("WSAECONNREFUSED", ConnectionRefused),
        ("WSAECONNRESET", ConnectionReset),
        ("ERROR_NOT_SAME_DEVICE", CrossesDevices),
        ("ERROR_POSSIBLE_DEADLOCK", Deadlock),
        ("ERROR_DIR_NOT_EMPTY", DirectoryNotEmpty),
        ("ERROR_CANT_RESOLVE_FILENAME", FilesystemLoop),
        ("ERROR_DISK_QUOTA_EXCEEDED", QuotaExceeded),
        ("WSAEDQUOT", QuotaExceeded),
        ("ERROR_FILE_TOO_LARGE", FileTooLarge),
        ("ERROR_HOST_UNREACHABLE", HostUnreachable),
        ("WSAEHOSTUNREACH", HostUnreachable),
        ("ERROR_INVALID_NAME", InvalidFilename),
        ("ERROR_BAD_PATHNAME", InvalidFilename),
        ("ERROR_FILENAME_EXCED_RANGE", InvalidFilename),
        ("ERROR_INVALID_PARAMETER", InvalidInput),
        ("WSAEINVAL", InvalidInput),
        ("ERROR_DIRECTORY_NOT_SUPPORTED", IsADirectory),
        ("WSAENETDOWN", NetworkDown),
        ("ERROR_NETWORK_UNREACHABLE", NetworkUnreachable),
        ("WSAENETUNREACH", NetworkUnreachable),
        ("ERROR_DIRECTORY", NotADirectory),
        ("WSAENOTCONN", NotConnected),
        ("ERROR_FILE_NOT_FOUND", NotFound),
        ("ERROR_PATH_NOT_FOUND", NotFound),
        ("ERROR_INVALID_DRIVE", NotFound),
        ("ERROR_BAD_NETPATH", NotFound),
        ("ERROR_BAD_NET_NAME", NotFound),
        ("ERROR_SEEK_ON_DEVICE", NotSeekable),
        ("ERROR_NOT_ENOUGH_MEMORY", OutOfMemory),
        ("ERROR_OUTOFMEMORY", OutOfMemory),
        ("ERROR_ACCESS_DENIED", PermissionDenied),
        ("WSAEACCES", PermissionDenied),
        ("ERROR_WRITE_PROTECT", ReadOnlyFilesystem),
        ("ERROR_BUSY", ResourceBusy),
        ("ERROR_DISK_FULL", StorageFull),
        ("ERROR_HANDLE_DISK_FULL", StorageFull),
        ("WAIT_TIMEOUT", TimedOut),
        ("WSAETIMEDOUT", TimedOut),
        ("ERROR_DRIVER_CANCEL_TIMEOUT", TimedOut),
        ("ERROR_OPERATION_ABORTED", TimedOut),
        ("ERROR_SERVICE_REQUEST_TIMEOUT", TimedOut),
        ("ERROR_COUNTER_TIMEOUT", TimedOut),
        ("ERROR_TIMEOUT", TimedOut),
        ("ERROR_RESOURCE_CALL_TIMED_OUT", TimedOut),
        ("ERROR_CTX_MODEM_RESPONSE_TIMEOUT", TimedOut),
        ("ERROR_CTX_CLIENT_QUERY_TIMEOUT", TimedOut),
        ("FRS_ERR_SYSVOL_POPULATE_TIMEOUT", TimedOut),
        ("ERROR_DS_TIMELIMIT_EXCEEDED", TimedOut),
        ("DNS_ERROR_RECORD_TIMED_OUT", TimedOut),
        ("ERROR_IPSEC_IKE_TIMED_OUT", TimedOut),
        ("ERROR_RUNLEVEL_SWITCH_TIMEOUT", TimedOut),
        ("ERROR_RUNLEVEL_SWITCH_AGENT_TIMEOUT", TimedOut),
        ("ERROR_TOO_MANY_LINKS", TooManyLinks),
        ("ERROR_CALL_NOT_IMPLEMENTED", Unsupported),
        ("WSAEWOULDBLOCK", WouldBlock),
    ]
};

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Get last error variable as a place, lazily allocating thread-local storage for it if
    /// necessary.
    fn last_error_place(&mut self) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
        let this = self.eval_context_mut();
        if let Some(errno_place) = this.active_thread_ref().last_error.as_ref() {
            interp_ok(errno_place.clone())
        } else {
            // Allocate new place, set initial value to 0.
            let errno_layout = this.machine.layouts.u32;
            let errno_place = this.allocate(errno_layout, MiriMemoryKind::Machine.into())?;
            this.write_scalar(Scalar::from_u32(0), &errno_place)?;
            this.active_thread_mut().last_error = Some(errno_place.clone());
            interp_ok(errno_place)
        }
    }

    /// Sets the last error variable.
    fn set_last_error(&mut self, err: impl Into<IoError>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let errno = match err.into() {
            HostError(err) => this.io_error_to_errnum(err)?,
            LibcError(name) => this.eval_libc(name),
            WindowsError(name) => this.eval_windows("c", name),
            Raw(val) => val,
        };
        let errno_place = this.last_error_place()?;
        this.write_scalar(errno, &errno_place)
    }

    /// Sets the last OS error and writes -1 to dest place.
    fn set_last_error_and_return(
        &mut self,
        err: impl Into<IoError>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.set_last_error(err)?;
        this.write_int(-1, dest)?;
        interp_ok(())
    }

    /// Sets the last OS error and return `-1` as a `i32`-typed Scalar
    fn set_last_error_and_return_i32(
        &mut self,
        err: impl Into<IoError>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.set_last_error(err)?;
        interp_ok(Scalar::from_i32(-1))
    }

    /// Sets the last OS error and return `-1` as a `i64`-typed Scalar
    fn set_last_error_and_return_i64(
        &mut self,
        err: impl Into<IoError>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        this.set_last_error(err)?;
        interp_ok(Scalar::from_i64(-1))
    }

    /// Gets the last error variable.
    fn get_last_error(&mut self) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        let errno_place = this.last_error_place()?;
        this.read_scalar(&errno_place)
    }

    /// This function converts host errors to target errors. It tries to produce the most similar OS
    /// error from the `std::io::ErrorKind` as a platform-specific errnum.
    fn io_error_to_errnum(&self, err: std::io::Error) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_ref();
        let target = &this.tcx.sess.target;

        if target.families.iter().any(|f| f == "unix") {
            for &(name, kind) in UNIX_IO_ERROR_TABLE {
                if err.kind() == kind {
                    return interp_ok(this.eval_libc(name));
                }
            }
            throw_unsup_format!("unsupported io error: {err}")
        } else if target.families.iter().any(|f| f == "windows") {
            for &(name, kind) in WINDOWS_IO_ERROR_TABLE {
                if err.kind() == kind {
                    return interp_ok(this.eval_windows("c", name));
                }
            }
            throw_unsup_format!("unsupported io error: {err}");
        } else {
            throw_unsup_format!(
                "converting io::Error into errnum is unsupported for OS {}",
                target.os
            )
        }
    }

    /// The inverse of `io_error_to_errnum`: it converts target errors to host errors.
    /// This is done in a best-effort way.
    #[expect(clippy::needless_return)]
    fn try_errnum_to_io_error(
        &self,
        target_errnum: Scalar,
    ) -> InterpResult<'tcx, Option<io::Error>> {
        let this = self.eval_context_ref();
        let target = &this.tcx.sess.target;
        if target.families.iter().any(|f| f == "unix") {
            let target_errnum = target_errnum.to_i32()?;
            // If the host is also unix, we try to translate the errno directly.
            // That lets us use `Error::from_raw_os_error`, which has a much better `Display`
            // impl than what we get by going through `ErrorKind`.
            #[cfg(unix)]
            for &(name, errno) in UNIX_ERRNO_TABLE {
                if target_errnum == this.eval_libc_i32(name) {
                    return interp_ok(Some(io::Error::from_raw_os_error(errno)));
                }
            }
            // For other hosts or other constants, we fall back to translating via `ErrorKind`.
            for &(name, kind) in UNIX_IO_ERROR_TABLE {
                if target_errnum == this.eval_libc_i32(name) {
                    return interp_ok(Some(kind.into()));
                }
            }
            return interp_ok(None);
        } else if target.families.iter().any(|f| f == "windows") {
            let target_errnum = target_errnum.to_u32()?;
            for &(name, kind) in WINDOWS_IO_ERROR_TABLE {
                if target_errnum == this.eval_windows("c", name).to_u32()? {
                    return interp_ok(Some(kind.into()));
                }
            }
            return interp_ok(None);
        } else {
            throw_unsup_format!(
                "converting errnum into io::Error is unsupported for OS {}",
                target.os
            )
        }
    }

    /// Helper function that consumes an `std::io::Result<T>` and returns an
    /// `InterpResult<'tcx,T>::Ok` instead. In case the result is an error, this function returns
    /// `Ok(-1)` and sets the last OS error accordingly.
    ///
    /// This function uses `T: From<i32>` instead of `i32` directly because some IO related
    /// functions return different integer types (like `read`, that returns an `i64`).
    fn try_unwrap_io_result<T: From<i32>>(
        &mut self,
        result: std::io::Result<T>,
    ) -> InterpResult<'tcx, T> {
        match result {
            Ok(ok) => interp_ok(ok),
            Err(e) => {
                self.eval_context_mut().set_last_error(e)?;
                interp_ok((-1).into())
            }
        }
    }
}
