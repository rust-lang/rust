use std::io;

use crate::*;

/// A representation of an IO error: either a libc error name,
/// or a host error.
#[derive(Debug)]
pub enum IoError {
    LibcError(&'static str),
    HostError(io::Error),
    Raw(Scalar),
}
pub use self::IoError::*;

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
// <https://github.com/rust-lang/rust/blob/master/library/std/src/sys/pal/unix/mod.rs>.
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
        ("EDQUOT", FilesystemQuotaExceeded),
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
        // The following have two valid options. We have both for the forwards mapping; only the
        // first one will be used for the backwards mapping.
        ("EPERM", PermissionDenied),
        ("EACCES", PermissionDenied),
        ("EWOULDBLOCK", WouldBlock),
        ("EAGAIN", WouldBlock),
    ]
};
// This mapping should match `decode_error_kind` in
// <https://github.com/rust-lang/rust/blob/master/library/std/src/sys/pal/windows/mod.rs>.
const WINDOWS_IO_ERROR_TABLE: &[(&str, std::io::ErrorKind)] = {
    use std::io::ErrorKind::*;
    // FIXME: this is still incomplete.
    &[
        ("ERROR_ACCESS_DENIED", PermissionDenied),
        ("ERROR_FILE_NOT_FOUND", NotFound),
        ("ERROR_INVALID_PARAMETER", InvalidInput),
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

    /// Gets the last error variable.
    fn get_last_error(&mut self) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        let errno_place = this.last_error_place()?;
        this.read_scalar(&errno_place)
    }

    /// This function tries to produce the most similar OS error from the `std::io::ErrorKind`
    /// as a platform-specific errnum.
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

    /// The inverse of `io_error_to_errnum`.
    #[allow(clippy::needless_return)]
    fn try_errnum_to_io_error(
        &self,
        errnum: Scalar,
    ) -> InterpResult<'tcx, Option<std::io::ErrorKind>> {
        let this = self.eval_context_ref();
        let target = &this.tcx.sess.target;
        if target.families.iter().any(|f| f == "unix") {
            let errnum = errnum.to_i32()?;
            for &(name, kind) in UNIX_IO_ERROR_TABLE {
                if errnum == this.eval_libc_i32(name) {
                    return interp_ok(Some(kind));
                }
            }
            return interp_ok(None);
        } else if target.families.iter().any(|f| f == "windows") {
            let errnum = errnum.to_u32()?;
            for &(name, kind) in WINDOWS_IO_ERROR_TABLE {
                if errnum == this.eval_windows("c", name).to_u32()? {
                    return interp_ok(Some(kind));
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
