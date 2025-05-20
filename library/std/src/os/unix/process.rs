//! Unix-specific extensions to primitives in the [`std::process`] module.
//!
//! [`std::process`]: crate::process

#![stable(feature = "rust1", since = "1.0.0")]

use cfg_if::cfg_if;

use crate::ffi::OsStr;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::path::Path;
use crate::sealed::Sealed;
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};
use crate::{io, process, sys};

cfg_if! {
    if #[cfg(any(target_os = "vxworks", target_os = "espidf", target_os = "horizon", target_os = "vita"))] {
        type UserId = u16;
        type GroupId = u16;
    } else if #[cfg(target_os = "nto")] {
        // Both IDs are signed, see `sys/target_nto.h` of the QNX Neutrino SDP.
        // Only positive values should be used, see e.g.
        // https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.lib_ref/topic/s/setuid.html
        type UserId = i32;
        type GroupId = i32;
    } else {
        type UserId = u32;
        type GroupId = u32;
    }
}

/// Unix-specific extensions to the [`process::Command`] builder.
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait CommandExt: Sealed {
    /// Sets the child process's user ID. This translates to a
    /// `setuid` call in the child process. Failure in the `setuid`
    /// call will cause the spawn to fail.
    ///
    /// # Notes
    ///
    /// This will also trigger a call to `setgroups(0, NULL)` in the child
    /// process if no groups have been specified.
    /// This removes supplementary groups that might have given the child
    /// unwanted permissions.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn uid(&mut self, id: UserId) -> &mut process::Command;

    /// Similar to `uid`, but sets the group ID of the child process. This has
    /// the same semantics as the `uid` field.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn gid(&mut self, id: GroupId) -> &mut process::Command;

    /// Sets the supplementary group IDs for the calling process. Translates to
    /// a `setgroups` call in the child process.
    #[unstable(feature = "setgroups", issue = "90747")]
    fn groups(&mut self, groups: &[GroupId]) -> &mut process::Command;

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
    /// like `malloc`, accessing environment variables through [`std::env`]
    /// or acquiring a mutex are not guaranteed to work (due to
    /// other threads perhaps still running when the `fork` was run).
    ///
    /// For further details refer to the [POSIX fork() specification]
    /// and the equivalent documentation for any targeted
    /// platform, especially the requirements around *async-signal-safety*.
    ///
    /// This also means that all resources such as file descriptors and
    /// memory-mapped regions got duplicated. It is your responsibility to make
    /// sure that the closure does not violate library invariants by making
    /// invalid use of these duplicates.
    ///
    /// Panicking in the closure is safe only if all the format arguments for the
    /// panic message can be safely formatted; this is because although
    /// `Command` calls [`std::panic::always_abort`](crate::panic::always_abort)
    /// before calling the pre_exec hook, panic will still try to format the
    /// panic message.
    ///
    /// When this closure is run, aspects such as the stdio file descriptors and
    /// working directory have successfully been changed, so output to these
    /// locations might not appear where intended.
    ///
    /// [POSIX fork() specification]:
    ///     https://pubs.opengroup.org/onlinepubs/9699919799/functions/fork.html
    /// [`std::env`]: mod@crate::env
    #[stable(feature = "process_pre_exec", since = "1.34.0")]
    unsafe fn pre_exec<F>(&mut self, f: F) -> &mut process::Command
    where
        F: FnMut() -> io::Result<()> + Send + Sync + 'static;

    /// Schedules a closure to be run just before the `exec` function is
    /// invoked.
    ///
    /// `before_exec` used to be a safe method, but it needs to be unsafe since the closure may only
    /// perform operations that are *async-signal-safe*. Hence it got deprecated in favor of the
    /// unsafe [`pre_exec`]. Meanwhile, Rust gained the ability to make an existing safe method
    /// fully unsafe in a new edition, which is how `before_exec` became `unsafe`. It still also
    /// remains deprecated; `pre_exec` should be used instead.
    ///
    /// [`pre_exec`]: CommandExt::pre_exec
    #[stable(feature = "process_exec", since = "1.15.0")]
    #[deprecated(since = "1.37.0", note = "should be unsafe, use `pre_exec` instead")]
    #[rustc_deprecated_safe_2024(audit_that = "the closure is async-signal-safe")]
    unsafe fn before_exec<F>(&mut self, f: F) -> &mut process::Command
    where
        F: FnMut() -> io::Result<()> + Send + Sync + 'static,
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
    /// `exec` not returning has the same implications as calling
    /// [`process::exit`] – no destructors on the current stack or any other
    /// thread’s stack will be run. Therefore, it is recommended to only call
    /// `exec` at a point where it is fine to not run any destructors. Note,
    /// that the `execvp` syscall independently guarantees that all memory is
    /// freed and all file descriptors with the `CLOEXEC` option (set by default
    /// on all file descriptors opened by the standard library) are closed.
    ///
    /// This function, unlike `spawn`, will **not** `fork` the process to create
    /// a new child. Like spawn, however, the default behavior for the stdio
    /// descriptors will be to inherit them from the current process.
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
    #[must_use]
    fn exec(&mut self) -> io::Error;

    /// Set executable argument
    ///
    /// Set the first process argument, `argv[0]`, to something other than the
    /// default executable path.
    #[stable(feature = "process_set_argv0", since = "1.45.0")]
    fn arg0<S>(&mut self, arg: S) -> &mut process::Command
    where
        S: AsRef<OsStr>;

    /// Sets the process group ID (PGID) of the child process. Equivalent to a
    /// `setpgid` call in the child process, but may be more efficient.
    ///
    /// Process groups determine which processes receive signals.
    ///
    /// # Examples
    ///
    /// Pressing Ctrl-C in a terminal will send SIGINT to all processes in
    /// the current foreground process group. By spawning the `sleep`
    /// subprocess in a new process group, it will not receive SIGINT from the
    /// terminal.
    ///
    /// The parent process could install a signal handler and manage the
    /// subprocess on its own terms.
    ///
    /// A process group ID of 0 will use the process ID as the PGID.
    ///
    /// ```no_run
    /// use std::process::Command;
    /// use std::os::unix::process::CommandExt;
    ///
    /// Command::new("sleep")
    ///     .arg("10")
    ///     .process_group(0)
    ///     .spawn()?
    ///     .wait()?;
    /// #
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    #[stable(feature = "process_set_process_group", since = "1.64.0")]
    fn process_group(&mut self, pgroup: i32) -> &mut process::Command;

    /// Set the root of the child process. This calls `chroot` in the child process before executing
    /// the command.
    ///
    /// This happens before changing to the directory specified with
    /// [`process::Command::current_dir`], and that directory will be relative to the new root.
    ///
    /// If no directory has been specified with [`process::Command::current_dir`], this will set the
    /// directory to `/`, to avoid leaving the current directory outside the chroot. (This is an
    /// intentional difference from the underlying `chroot` system call.)
    #[unstable(feature = "process_chroot", issue = "141298")]
    fn chroot<P: AsRef<Path>>(&mut self, dir: P) -> &mut process::Command;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl CommandExt for process::Command {
    fn uid(&mut self, id: UserId) -> &mut process::Command {
        self.as_inner_mut().uid(id);
        self
    }

    fn gid(&mut self, id: GroupId) -> &mut process::Command {
        self.as_inner_mut().gid(id);
        self
    }

    fn groups(&mut self, groups: &[GroupId]) -> &mut process::Command {
        self.as_inner_mut().groups(groups);
        self
    }

    unsafe fn pre_exec<F>(&mut self, f: F) -> &mut process::Command
    where
        F: FnMut() -> io::Result<()> + Send + Sync + 'static,
    {
        self.as_inner_mut().pre_exec(Box::new(f));
        self
    }

    fn exec(&mut self) -> io::Error {
        // NOTE: This may *not* be safe to call after `libc::fork`, because it
        // may allocate. That may be worth fixing at some point in the future.
        self.as_inner_mut().exec(sys::process::Stdio::Inherit)
    }

    fn arg0<S>(&mut self, arg: S) -> &mut process::Command
    where
        S: AsRef<OsStr>,
    {
        self.as_inner_mut().set_arg_0(arg.as_ref());
        self
    }

    fn process_group(&mut self, pgroup: i32) -> &mut process::Command {
        self.as_inner_mut().pgroup(pgroup);
        self
    }

    fn chroot<P: AsRef<Path>>(&mut self, dir: P) -> &mut process::Command {
        self.as_inner_mut().chroot(dir.as_ref());
        self
    }
}

/// Unix-specific extensions to [`process::ExitStatus`] and
/// [`ExitStatusError`](process::ExitStatusError).
///
/// On Unix, `ExitStatus` **does not necessarily represent an exit status**, as
/// passed to the `_exit` system call or returned by
/// [`ExitStatus::code()`](crate::process::ExitStatus::code).  It represents **any wait status**
/// as returned by one of the `wait` family of system
/// calls.
///
/// A Unix wait status (a Rust `ExitStatus`) can represent a Unix exit status, but can also
/// represent other kinds of process event.
///
/// This trait is sealed: it cannot be implemented outside the standard library.
/// This is so that future additional methods are not breaking changes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExitStatusExt: Sealed {
    /// Creates a new `ExitStatus` or `ExitStatusError` from the raw underlying integer status
    /// value from `wait`
    ///
    /// The value should be a **wait status, not an exit status**.
    ///
    /// # Panics
    ///
    /// Panics on an attempt to make an `ExitStatusError` from a wait status of `0`.
    ///
    /// Making an `ExitStatus` always succeeds and never panics.
    #[stable(feature = "exit_status_from", since = "1.12.0")]
    fn from_raw(raw: i32) -> Self;

    /// If the process was terminated by a signal, returns that signal.
    ///
    /// In other words, if `WIFSIGNALED`, this returns `WTERMSIG`.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn signal(&self) -> Option<i32>;

    /// If the process was terminated by a signal, says whether it dumped core.
    #[stable(feature = "unix_process_wait_more", since = "1.58.0")]
    fn core_dumped(&self) -> bool;

    /// If the process was stopped by a signal, returns that signal.
    ///
    /// In other words, if `WIFSTOPPED`, this returns `WSTOPSIG`.  This is only possible if the status came from
    /// a `wait` system call which was passed `WUNTRACED`, and was then converted into an `ExitStatus`.
    #[stable(feature = "unix_process_wait_more", since = "1.58.0")]
    fn stopped_signal(&self) -> Option<i32>;

    /// Whether the process was continued from a stopped status.
    ///
    /// Ie, `WIFCONTINUED`.  This is only possible if the status came from a `wait` system call
    /// which was passed `WCONTINUED`, and was then converted into an `ExitStatus`.
    #[stable(feature = "unix_process_wait_more", since = "1.58.0")]
    fn continued(&self) -> bool;

    /// Returns the underlying raw `wait` status.
    ///
    /// The returned integer is a **wait status, not an exit status**.
    #[stable(feature = "unix_process_wait_more", since = "1.58.0")]
    fn into_raw(self) -> i32;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ExitStatusExt for process::ExitStatus {
    fn from_raw(raw: i32) -> Self {
        process::ExitStatus::from_inner(From::from(raw))
    }

    fn signal(&self) -> Option<i32> {
        self.as_inner().signal()
    }

    fn core_dumped(&self) -> bool {
        self.as_inner().core_dumped()
    }

    fn stopped_signal(&self) -> Option<i32> {
        self.as_inner().stopped_signal()
    }

    fn continued(&self) -> bool {
        self.as_inner().continued()
    }

    fn into_raw(self) -> i32 {
        self.as_inner().into_raw().into()
    }
}

#[unstable(feature = "exit_status_error", issue = "84908")]
impl ExitStatusExt for process::ExitStatusError {
    fn from_raw(raw: i32) -> Self {
        process::ExitStatus::from_raw(raw)
            .exit_ok()
            .expect_err("<ExitStatusError as ExitStatusExt>::from_raw(0) but zero is not an error")
    }

    fn signal(&self) -> Option<i32> {
        self.into_status().signal()
    }

    fn core_dumped(&self) -> bool {
        self.into_status().core_dumped()
    }

    fn stopped_signal(&self) -> Option<i32> {
        self.into_status().stopped_signal()
    }

    fn continued(&self) -> bool {
        self.into_status().continued()
    }

    fn into_raw(self) -> i32 {
        self.into_status().into_raw()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawFd for process::Stdio {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> process::Stdio {
        let fd = sys::fd::FileDesc::from_raw_fd(fd);
        let io = sys::process::Stdio::Fd(fd);
        process::Stdio::from_inner(io)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<OwnedFd> for process::Stdio {
    /// Takes ownership of a file descriptor and returns a [`Stdio`](process::Stdio)
    /// that can attach a stream to it.
    #[inline]
    fn from(fd: OwnedFd) -> process::Stdio {
        let fd = sys::fd::FileDesc::from_inner(fd);
        let io = sys::process::Stdio::Fd(fd);
        process::Stdio::from_inner(io)
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdin {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().as_raw_fd()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdout {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().as_raw_fd()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStderr {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().as_raw_fd()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdin {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_inner().into_raw_fd()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdout {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_inner().into_raw_fd()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStderr {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_inner().into_raw_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for crate::process::ChildStdin {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::process::ChildStdin> for OwnedFd {
    /// Takes ownership of a [`ChildStdin`](crate::process::ChildStdin)'s file descriptor.
    #[inline]
    fn from(child_stdin: crate::process::ChildStdin) -> OwnedFd {
        child_stdin.into_inner().into_inner().into_inner()
    }
}

/// Creates a `ChildStdin` from the provided `OwnedFd`.
///
/// The provided file descriptor must point to a pipe
/// with the `CLOEXEC` flag set.
#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedFd> for process::ChildStdin {
    #[inline]
    fn from(fd: OwnedFd) -> process::ChildStdin {
        let fd = sys::fd::FileDesc::from_inner(fd);
        let pipe = sys::pipe::AnonPipe::from_inner(fd);
        process::ChildStdin::from_inner(pipe)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for crate::process::ChildStdout {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::process::ChildStdout> for OwnedFd {
    /// Takes ownership of a [`ChildStdout`](crate::process::ChildStdout)'s file descriptor.
    #[inline]
    fn from(child_stdout: crate::process::ChildStdout) -> OwnedFd {
        child_stdout.into_inner().into_inner().into_inner()
    }
}

/// Creates a `ChildStdout` from the provided `OwnedFd`.
///
/// The provided file descriptor must point to a pipe
/// with the `CLOEXEC` flag set.
#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedFd> for process::ChildStdout {
    #[inline]
    fn from(fd: OwnedFd) -> process::ChildStdout {
        let fd = sys::fd::FileDesc::from_inner(fd);
        let pipe = sys::pipe::AnonPipe::from_inner(fd);
        process::ChildStdout::from_inner(pipe)
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl AsFd for crate::process::ChildStderr {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[stable(feature = "io_safety", since = "1.63.0")]
impl From<crate::process::ChildStderr> for OwnedFd {
    /// Takes ownership of a [`ChildStderr`](crate::process::ChildStderr)'s file descriptor.
    #[inline]
    fn from(child_stderr: crate::process::ChildStderr) -> OwnedFd {
        child_stderr.into_inner().into_inner().into_inner()
    }
}

/// Creates a `ChildStderr` from the provided `OwnedFd`.
///
/// The provided file descriptor must point to a pipe
/// with the `CLOEXEC` flag set.
#[stable(feature = "child_stream_from_fd", since = "1.74.0")]
impl From<OwnedFd> for process::ChildStderr {
    #[inline]
    fn from(fd: OwnedFd) -> process::ChildStderr {
        let fd = sys::fd::FileDesc::from_inner(fd);
        let pipe = sys::pipe::AnonPipe::from_inner(fd);
        process::ChildStderr::from_inner(pipe)
    }
}

/// Returns the OS-assigned process identifier associated with this process's parent.
#[must_use]
#[stable(feature = "unix_ppid", since = "1.27.0")]
pub fn parent_id() -> u32 {
    crate::sys::os::getppid()
}
