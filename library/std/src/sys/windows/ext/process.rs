//! Extensions to `std::process` for Windows.

#![stable(feature = "process_extensions", since = "1.2.0")]

use crate::ffi::OsStr;
use crate::io;
use crate::os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle, RawHandle};
use crate::process;
use crate::sys;
#[unstable(feature = "windows_raw_cmdline", issue = "74549")]
pub use crate::sys::process::{Arg, Problem, RawArg};
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};
use core::convert::TryFrom;

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
#[stable(feature = "windows_process_extensions", since = "1.16.0")]
pub trait CommandExt {
    /// Sets the [process creation flags][1] to be passed to `CreateProcess`.
    ///
    /// These will always be ORed with `CREATE_UNICODE_ENVIRONMENT`.
    ///
    /// [1]: https://docs.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
    #[stable(feature = "windows_process_extensions", since = "1.16.0")]
    fn creation_flags(&mut self, flags: u32) -> &mut process::Command;

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

/// Traits for handling a sized command.
// FIXME: This really should be somewhere else, since it will be duplicated for unix. sys_common? I have no idea.
// The implementations should apply to unix, but describing it to the type system is another thing.
#[unstable(feature = "command_sized", issue = "74549")]
pub trait CommandSized: core::marker::Sized {
    /// Possibly pass an argument.
    /// Returns an error if the size of the arguments would overflow the command line. The error contains the reason the remaining arguments could not be added.
    fn maybe_arg(&mut self, arg: impl Arg) -> io::Result<&mut Self>;
    /// Possibly pass many arguments. 
    /// Returns an error if the size of the arguments would overflow the command line. The error contains the number of arguments added as well as the reason the remaining arguments could not be added.
    fn maybe_args(
        &mut self,
        args: &mut impl Iterator<Item = impl Arg>,
    ) -> Result<&mut Self, (usize, io::Error)>;
    /// Build multiple commands to consume all arguments.
    /// Returns an error if the size of an argument would overflow the command line. The error contains the reason the remaining arguments could not be added.
    fn xargs<I, S, A>(program: S, args: &mut I, before: Vec<A>, after: Vec<A>) -> io::Result<Vec<Self>>
    where
        I: Iterator<Item = A>,
        S: AsRef<OsStr> + Copy,
        A: Arg;
}

#[unstable(feature = "command_sized", issue = "74549")]
impl CommandSized for process::Command {
    fn maybe_arg(&mut self, arg: impl Arg) -> io::Result<&mut Self> {
        self.as_inner_mut().maybe_arg_ext(arg)?;
        Ok(self)
    }
    fn maybe_args(
        &mut self,
        args: &mut impl Iterator<Item = impl Arg>,
    ) -> Result<&mut Self, (usize, io::Error)> {
        let mut count: usize = 0;
        for arg in args {
            if let Err(err) = self.as_inner_mut().maybe_arg_ext(arg) {
                return Err((count, err));
            }
            count += 1;
        }
        Ok(self)
    }
    fn xargs<I, S, A>(program: S, args: &mut I, before: Vec<A>, after: Vec<A>) -> io::Result<Vec<Self>>
    where
        I: Iterator<Item = A>,
        S: AsRef<OsStr> + Copy,
        A: Arg,
    {
        let mut ret = Vec::new();
        let mut cmd = Self::new(program);
        let mut fresh: bool = true;

        // This performs a nul check.
        let tail_size: usize = after
            .iter()
            .map(|x| Arg::arg_size(x, false))
            .collect::<Result<Vec<_>, Problem>>()?
            .iter()
            .sum();

        if let Err(_) = isize::try_from(tail_size) {
            return Err(Problem::Oversized.into());
        }

        cmd.args_ext(&before);
        if cmd.as_inner_mut().available_size(false)? < (tail_size as isize) {
            return Err(Problem::Oversized.into());
        }

        for arg in args {
            let size = arg.arg_size(false)?;
            // Negative case is catched outside of loop.
            if (cmd.as_inner_mut().available_size(false)? as usize) < (size + tail_size) {
                if fresh {
                    return Err(Problem::Oversized.into());
                }
                cmd.args_ext(&after);
                ret.push(cmd);
                cmd = Self::new(program);
                cmd.args_ext(&before);
            }
            cmd.maybe_arg(arg)?;
            fresh = false;
        }
        cmd.args_ext(&after);
        ret.push(cmd);
        Ok(ret)
    }
}
