//! Redox-specific extensions to primitives in the `std::process` module.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::io;
use crate::os::unix::io::{FromRawFd, RawFd, AsRawFd, IntoRawFd};
use crate::process;
use crate::sys;
use crate::sys_common::{AsInnerMut, AsInner, FromInner, IntoInner};

/// Redox-specific extensions to the [`process::Command`] builder,
///
/// [`process::Command`]: ../../../../std/process/struct.Command.html
#[stable(feature = "rust1", since = "1.0.0")]
pub trait CommandExt {
    /// Sets the child process's user ID. This translates to a
    /// `setuid` call in the child process. Failure in the `setuid`
    /// call will cause the spawn to fail.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn uid(&mut self, id: u32) -> &mut process::Command;

    /// Similar to `uid`, but sets the group ID of the child process. This has
    /// the same semantics as the `uid` field.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn gid(&mut self, id: u32) -> &mut process::Command;

    /// Schedules a closure to be run just before the `exec` function is
    /// invoked.
    ///
    /// The closure is allowed to return an I/O error whose OS error code will
    /// be communicated back to the parent and returned as an error from when
    /// the spawn was requested.
    ///
    /// Multiple closures can be registered and they will be called in order of
    /// their registration. If a closure returns `Err` then no further closures
    /// will be called and the spawn operation will immediately return with a
    /// failure.
    ///
    /// # Notes and Safety
    ///
    /// This closure will be run in the context of the child process after a
    /// `fork`. This primarily means that any modifications made to memory on
    /// behalf of this closure will **not** be visible to the parent process.
    /// This is often a very constrained environment where normal operations
    /// like `malloc` or acquiring a mutex are not guaranteed to work (due to
    /// other threads perhaps still running when the `fork` was run).
    ///
    /// This also means that all resources such as file descriptors and
    /// memory-mapped regions got duplicated. It is your responsibility to make
    /// sure that the closure does not violate library invariants by making
    /// invalid use of these duplicates.
    ///
    /// When this closure is run, aspects such as the stdio file descriptors and
    /// working directory have successfully been changed, so output to these
    /// locations may not appear where intended.
    #[stable(feature = "process_pre_exec", since = "1.34.0")]
    unsafe fn pre_exec<F>(&mut self, f: F) -> &mut process::Command
        where F: FnMut() -> io::Result<()> + Send + Sync + 'static;

    /// Schedules a closure to be run just before the `exec` function is
    /// invoked.
    ///
    /// This method is stable and usable, but it should be unsafe. To fix
    /// that, it got deprecated in favor of the unsafe [`pre_exec`].
    ///
    /// [`pre_exec`]: #tymethod.pre_exec
    #[stable(feature = "process_exec", since = "1.15.0")]
    #[rustc_deprecated(since = "1.37.0", reason = "should be unsafe, use `pre_exec` instead")]
    fn before_exec<F>(&mut self, f: F) -> &mut process::Command
        where F: FnMut() -> io::Result<()> + Send + Sync + 'static
    {
        unsafe { self.pre_exec(f) }
    }

    /// Performs all the required setup by this `Command`, followed by calling
    /// the `execvp` syscall.
    ///
    /// On success this function will not return, and otherwise it will return
    /// an error indicating why the exec (or another part of the setup of the
    /// `Command`) failed.
    ///
    /// This function, unlike `spawn`, will **not** `fork` the process to create
    /// a new child. Like spawn, however, the default behavior for the stdio
    /// descriptors will be to inherited from the current process.
    ///
    /// # Notes
    ///
    /// The process may be in a "broken state" if this function returns in
    /// error. For example the working directory, environment variables, signal
    /// handling settings, various user/group information, or aspects of stdio
    /// file descriptors may have changed. If a "transactional spawn" is
    /// required to gracefully handle errors it is recommended to use the
    /// cross-platform `spawn` instead.
    #[stable(feature = "process_exec2", since = "1.9.0")]
    fn exec(&mut self) -> io::Error;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl CommandExt for process::Command {
    fn uid(&mut self, id: u32) -> &mut process::Command {
        self.as_inner_mut().uid(id);
        self
    }

    fn gid(&mut self, id: u32) -> &mut process::Command {
        self.as_inner_mut().gid(id);
        self
    }

    unsafe fn pre_exec<F>(&mut self, f: F) -> &mut process::Command
        where F: FnMut() -> io::Result<()> + Send + Sync + 'static
    {
        self.as_inner_mut().pre_exec(Box::new(f));
        self
    }

    fn exec(&mut self) -> io::Error {
        self.as_inner_mut().exec(sys::process::Stdio::Inherit)
    }
}

/// Redox-specific extensions to [`process::ExitStatus`].
///
/// [`process::ExitStatus`]: ../../../../std/process/struct.ExitStatus.html
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExitStatusExt {
    /// Creates a new `ExitStatus` from the raw underlying `i32` return value of
    /// a process.
    #[stable(feature = "exit_status_from", since = "1.12.0")]
    fn from_raw(raw: i32) -> Self;

    /// If the process was terminated by a signal, returns that signal.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn signal(&self) -> Option<i32>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ExitStatusExt for process::ExitStatus {
    fn from_raw(raw: i32) -> Self {
        process::ExitStatus::from_inner(From::from(raw))
    }

    fn signal(&self) -> Option<i32> {
        self.as_inner().signal()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawFd for process::Stdio {
    unsafe fn from_raw_fd(fd: RawFd) -> process::Stdio {
        let fd = sys::fd::FileDesc::new(fd);
        let io = sys::process::Stdio::Fd(fd);
        process::Stdio::from_inner(io)
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdin {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdout {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStderr {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().fd().raw()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdin {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdout {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStderr {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_fd().into_raw()
    }
}
