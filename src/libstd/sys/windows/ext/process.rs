//! Extensions to `std::process` for Windows.

#![stable(feature = "process_extensions", since = "1.2.0")]

use crate::os::windows::io::{FromRawHandle, RawHandle, AsRawHandle, IntoRawHandle};
use crate::process;
use crate::sys;
use crate::sys_common::{AsInnerMut, AsInner, FromInner, IntoInner};

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawHandle for process::Stdio {
    unsafe fn from_raw_handle(handle: RawHandle) -> process::Stdio {
        let handle = sys::handle::Handle::new(handle as *mut _);
        let io = sys::process::Stdio::Handle(handle);
        process::Stdio::from_inner(io)
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::Child {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::Child {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStdin {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStdout {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStderr {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::ChildStdin {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::ChildStdout {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for process::ChildStderr {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

/// Windows-specific extensions to [`process::ExitStatus`].
///
/// [`process::ExitStatus`]: ../../../../std/process/struct.ExitStatus.html
#[stable(feature = "exit_status_from", since = "1.12.0")]
pub trait ExitStatusExt {
    /// Creates a new `ExitStatus` from the raw underlying `u32` return value of
    /// a process.
    #[stable(feature = "exit_status_from", since = "1.12.0")]
    fn from_raw(raw: u32) -> Self;
}

#[stable(feature = "exit_status_from", since = "1.12.0")]
impl ExitStatusExt for process::ExitStatus {
    fn from_raw(raw: u32) -> Self {
        process::ExitStatus::from_inner(From::from(raw))
    }
}

/// Windows-specific extensions to the [`process::Command`] builder.
///
/// [`process::Command`]: ../../../../std/process/struct.Command.html
#[stable(feature = "windows_process_extensions", since = "1.16.0")]
pub trait CommandExt {
    /// Sets the [process creation flags][1] to be passed to `CreateProcess`.
    ///
    /// These will always be ORed with `CREATE_UNICODE_ENVIRONMENT`.
    ///
    /// [1]: https://msdn.microsoft.com/en-us/library/windows/desktop/ms684863(v=vs.85).aspx
    #[stable(feature = "windows_process_extensions", since = "1.16.0")]
    fn creation_flags(&mut self, flags: u32) -> &mut process::Command;
}

#[stable(feature = "windows_process_extensions", since = "1.16.0")]
impl CommandExt for process::Command {
    fn creation_flags(&mut self, flags: u32) -> &mut process::Command {
        self.as_inner_mut().creation_flags(flags);
        self
    }
}
