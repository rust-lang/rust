//! Extensions to `std::process` for Windows.

#![stable(feature = "process_extensions", since = "1.2.0")]

use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle, RawHandle};
use crate::process;
use crate::sealed::Sealed;
use crate::sys;
use crate::os::windows::ffi::OsStrExt;
#[unstable(feature = "windows_raw_cmdline", issue = "74549")]
pub use crate::sys_common::process_ext::{Arg, Problem};
use crate::sys_common::{process_ext, AsInner, AsInnerMut, FromInner, IntoInner};
use core::convert::TryFrom;

/// Argument type with no escaping.
#[unstable(feature = "windows_raw_cmdline", issue = "74549")]
pub struct RawArg<'a>(&'a OsStr);

// FIXME: Inhibiting doc on non-Windows due to mismatching trait methods.
#[cfg(any(windows))]
#[doc(cfg(windows))]
#[unstable(feature = "windows_raw_cmdline", issue = "74549")]
impl Arg for RawArg<'_> {
    fn append_to(&self, cmd: &mut Vec<u16>, _fq: bool) -> Result<usize, Problem> {
        cmd.extend(self.0.encode_wide());
        self.arg_size(_fq)
    }
    fn arg_size(&self, _: bool) -> Result<usize, Problem> {
        Ok(self.0.encode_wide().count() + 1)
    }
    fn to_os_string(&self) -> OsString {
        OsStr::to_os_string(&(self.0))
    }
}

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
    #[inline]
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
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStdout {
    #[inline]
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawHandle for process::ChildStderr {
    #[inline]
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
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "exit_status_from", since = "1.12.0")]
pub trait ExitStatusExt: Sealed {
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
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "windows_process_extensions", since = "1.16.0")]
pub trait CommandExt: Sealed {
    /// Sets the [process creation flags][1] to be passed to `CreateProcess`.
    ///
    /// These will always be ORed with `CREATE_UNICODE_ENVIRONMENT`.
    ///
    /// [1]: https://docs.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
    #[stable(feature = "windows_process_extensions", since = "1.16.0")]
    fn creation_flags(&mut self, flags: u32) -> &mut process::Command;

    /// Forces all arguments to be wrapped in quote (`"`) characters.
    ///
    /// This is useful for passing arguments to [MSYS2/Cygwin][1] based
    /// executables: these programs will expand unquoted arguments containing
    /// wildcard characters (`?` and `*`) by searching for any file paths
    /// matching the wildcard pattern.
    ///
    /// Adding quotes has no effect when passing arguments to programs
    /// that use [msvcrt][2]. This includes programs built with both
    /// MinGW and MSVC.
    ///
    /// [1]: <https://github.com/msys2/MSYS2-packages/issues/2176>
    /// [2]: <https://msdn.microsoft.com/en-us/library/17w5ykft.aspx>
    #[unstable(feature = "windows_process_extensions_force_quotes", issue = "82227")]
    fn force_quotes(&mut self, enabled: bool) -> &mut process::Command;
    /// Pass an argument with custom escape rules.
    #[unstable(feature = "windows_raw_cmdline", issue = "74549")]
    fn arg_ext(&mut self, arg: impl Arg) -> &mut process::Command;

    /// Pass arguments with custom escape rules.
    #[unstable(feature = "windows_raw_cmdline", issue = "74549")]
    fn args_ext(&mut self, args: impl IntoIterator<Item = impl Arg>) -> &mut process::Command;
}

#[stable(feature = "windows_process_extensions", since = "1.16.0")]
impl CommandExt for process::Command {
    fn creation_flags(&mut self, flags: u32) -> &mut process::Command {
        self.as_inner_mut().creation_flags(flags);
        self
    }

    fn force_quotes(&mut self, enabled: bool) -> &mut process::Command {
        self.as_inner_mut().force_quotes(enabled);
        self
    }

    fn arg_ext(&mut self, arg: impl Arg) -> &mut process::Command {
        self.as_inner_mut().arg_ext(arg);
        self
    }

    fn args_ext(&mut self, args: impl IntoIterator<Item = impl Arg>) -> &mut process::Command {
        for arg in args {
            self.arg_ext(arg);
        }
        self
    }
}

// FIXME: export maybe_arg_ext so the macro doesn't explicitly reach for as_inner_mut()
#[unstable(feature = "command_sized", issue = "74549")]
#[cfg(windows)] // doc hack
impl process_ext::CommandSized for process::Command {
    impl_command_sized! { marg  sys::process::Command::maybe_arg_ext }
    impl_command_sized! { margs sys::process::Command::maybe_arg_ext }
    impl_command_sized! { xargs process::Command::args_ext }
}
