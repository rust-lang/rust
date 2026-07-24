//! Unix-specific extensions to primitives in the [`std::process`] module.
//!
//! [`std::process`]: crate::process

#![stable(feature = "rust1", since = "1.0.0")]

use crate::ffi::OsStr;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::path::Path;
#[cfg(doc)]
use crate::process::{ExitStatus, ExitStatusError};
use crate::sys::process::ChildPipe;
use crate::sys::{AsInner, AsInnerMut, FromInner, IntoInner};
use crate::{io, process, sys};

cfg_select! {
    any(target_os = "vxworks", target_os = "espidf", target_os = "horizon", target_os = "vita") => {
        type UserId = u16;
        type GroupId = u16;
    }
    any(target_os = "nto", target_os = "qnx") => {
        // Both IDs are signed, see `sys/target_nto.h` of the QNX SDP.
        // Only positive values should be used, see e.g.
        // https://www.qnx.com/developers/docs/7.1/com.qnx.doc.neutrino.lib_ref/topic/s/setuid.html
        type UserId = i32;
        type GroupId = i32;
    }
    _ => {
        type UserId = u32;
        type GroupId = u32;
    }
}

/// Unix-specific extensions to the [`process::Command`] builder.
#[stable(feature = "rust1", since = "1.0.0")]
pub impl(self) trait CommandExt {
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
    /// Note that the list of allocating functions includes [`Error::new`] and
    /// [`Error::other`]. To signal a non-trivial error, prefer [`panic!`].
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
    ///     https://pubs.opengroup.org/onlinepubs/9799919799/functions/fork.html
    /// [`std::env`]: mod@crate::env
    /// [`Error::new`]: ../../../io/struct.Error.html#method.new
    /// [`Error::other`]: ../../../io/struct.Error.html#method.other
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

    #[unstable(feature = "process_setsid", issue = "105376")]
    fn setsid(&mut self, setsid: bool) -> &mut process::Command;

    /// Pass a file descriptor to a child process.
    ///
    /// `old_fd` is an open file descriptor in the parent process. This fd will be duplicated in the
    /// child process and associated with the fd number `new_fd`.
    ///
    /// Getting this right is tricky. It is recommended to provide further information to the child
    /// process by some other mechanism. This could be an argument confirming file descriptors that
    /// the child can use, device/inode numbers to allow for sanity checks, or something similar.
    ///
    /// If `old_fd` is an open file descriptor in the child process (e.g. if multiple parent fds are being
    /// mapped to the same child one) and closing it would produce one or more errors,
    /// those errors will be lost when this function is called. See
    /// [`man 2 dup`](https://www.man7.org/linux/man-pages/man2/dup.2.html#NOTES) for more information.
    ///
    /// ```
    /// #![feature(command_pass_fds)]
    ///
    /// use std::process::{Command, Stdio};
    /// use std::os::unix::process::CommandExt;
    /// use std::io::{self, Write};
    ///
    /// # fn main() -> io::Result<()> {
    /// let (pipe_reader, mut pipe_writer) = io::pipe()?;
    ///
    /// let fd_num = 123;
    ///
    /// let mut cmd = Command::new("cat");
    /// cmd.arg(format!("/dev/fd/{fd_num}")).stdout(Stdio::piped()).fd(fd_num, pipe_reader);
    ///
    /// let mut child = cmd.spawn()?;
    /// let mut stdout = child.stdout.take().unwrap();
    ///
    /// pipe_writer.write_all(b"Hello, world!")?;
    /// drop(pipe_writer);
    ///
    /// child.wait()?;
    /// assert_eq!(io::read_to_string(&mut stdout)?, "Hello, world!");
    ///
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// If this method is called multiple times with the same `new_fd`, all but one file descriptor
    /// will be lost.
    ///
    /// ```
    /// #![feature(command_pass_fds)]
    ///
    /// use std::process::{Command, Stdio};
    /// use std::os::unix::process::CommandExt;
    /// use std::io::{self, Write};
    ///
    /// # fn main() -> io::Result<()> {
    /// let (pipe_reader1, mut pipe_writer1) = io::pipe()?;
    /// let (pipe_reader2, mut pipe_writer2) = io::pipe()?;
    ///
    /// let fd_num = 123;
    ///
    /// let mut cmd = Command::new("cat");
    /// cmd.arg(format!("/dev/fd/{fd_num}"))
    ///     .stdout(Stdio::piped())
    ///     .fd(fd_num, pipe_reader1)
    ///     .fd(fd_num, pipe_reader2);
    ///
    /// pipe_writer1.write_all(b"Hello from pipe 1!")?;
    /// drop(pipe_writer1);
    ///
    /// pipe_writer2.write_all(b"Hello from pipe 2!")?;
    /// drop(pipe_writer2);
    ///
    /// let mut child = cmd.spawn()?;
    /// let mut stdout = child.stdout.take().unwrap();
    ///
    /// child.wait()?;
    /// assert_eq!(io::read_to_string(&mut stdout)?, "Hello from pipe 2!");
    ///
    /// # Ok(())
    /// # }
    /// ```
    #[unstable(feature = "command_pass_fds", issue = "144989")]
    fn fd(&mut self, new_fd: RawFd, old_fd: impl Into<OwnedFd>) -> &mut Self;
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

    fn setsid(&mut self, setsid: bool) -> &mut process::Command {
        self.as_inner_mut().setsid(setsid);
        self
    }

    fn fd(&mut self, new_fd: RawFd, old_fd: impl Into<OwnedFd>) -> &mut Self {
        self.as_inner_mut().fd(old_fd.into(), new_fd);
        self
    }
}

/// Unix-specific extensions to [`ExitStatus`] and [`ExitStatusError`].
///
/// On Unix, [`ExitStatus`] **does not necessarily represent an exit status**, as
/// passed to the `_exit` system call or returned by
/// [`ExitStatus::code()`](ExitStatus::code).  It represents **any wait status**
/// as returned by one of the [`wait`] family of system
/// calls.
///
/// A Unix wait status (a Rust [`ExitStatus`]) can represent a Unix exit status, but can also
/// represent other kinds of process event.
///
/// [`wait`]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/wait.html
#[stable(feature = "rust1", since = "1.0.0")]
pub impl(self) trait ExitStatusExt {
    /// Creates a new [`ExitStatus`] or [`ExitStatusError`] from the raw underlying integer status
    /// value from [`wait`].
    ///
    /// The value should be a **wait status, not an exit status**.
    ///
    /// # Example
    ///
    /// A signal-terminated [`wait`] status carries the signal number, which [`ExitStatus::signal`]
    /// recovers using the platform's [`WTERMSIG`][`wait`] macro. Note that the bit layout of a
    /// wait status is **not** specified by POSIX and is platform-specific. By convention on most
    /// Unix platforms, the signal number occupies the low 7 bits with the exit-code byte left
    /// zero, so a bare signal number between 1 and 126 is treated as a signal-terminated wait
    /// status. The following example relies on that convention and is therefore not guaranteed to
    /// hold on every target:
    ///
    /// ```
    /// # if cfg!(target_os = "fuchsia") { return; }
    /// use std::os::unix::process::ExitStatusExt;
    /// use std::process::ExitStatus;
    ///
    /// let signal = 15; // SIGTERM
    /// assert!(signal > 0 && signal < 0x7f, "not a valid Unix termination signal: {signal}");
    ///
    /// let status = ExitStatus::from_raw(signal);
    /// assert!(!status.success());
    /// assert_eq!(status.code(), None);
    /// assert_eq!(status.signal(), Some(15));
    /// ```
    ///
    /// Generating an [`ExitStatus`] with a given exit code (0-255) is system-dependent.
    /// The value returned by [`ExitStatus::code`] is specified to come from applying the
    /// [`WEXITSTATUS`][`wait`] macro, but there is no POSIX-specified constructor and the bit
    /// layout is left unspecified. By near-universal convention every Unix libc stores the
    /// 8-bit exit code in bits 8..16, so a status built with `(code & 0xff) << 8` will usually
    /// round-trip back to the original exit code:
    ///
    /// ```
    /// # if cfg!(target_os = "fuchsia") { return; }
    /// use std::os::unix::process::ExitStatusExt;
    /// use std::process::ExitStatus;
    ///
    /// let code = 41;
    /// let status = ExitStatus::from_raw((code & 0xff) << 8);
    /// assert_eq!(status.code(), Some(41));
    /// assert!(!status.success());
    /// ```
    ///
    /// # Panics
    ///
    /// - `ExitStatusError::from_raw` panics on an attempt to make an [`ExitStatusError`] from a
    ///    [`wait`] status of `0`.
    /// - `ExitStatus::from_raw` always succeeds and never panics.
    ///
    /// [`wait`]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/wait.html
    #[stable(feature = "exit_status_from", since = "1.12.0")]
    fn from_raw(raw: i32) -> Self;

    /// If the process was terminated by a signal, returns that signal.
    ///
    /// In other words, if [`WIFSIGNALED`][`wait`], this returns [`WTERMSIG`][`wait`]. For such a status,
    /// [`ExitStatus::code`] returns `None`:
    ///
    /// ```
    /// # if cfg!(target_os = "fuchsia") { return; }
    /// use std::os::unix::process::ExitStatusExt;
    /// use std::process::ExitStatus;
    ///
    /// let sigterm = 15;
    /// let status = ExitStatus::from_raw(sigterm);
    /// assert_eq!(status.code(), None);
    /// assert_eq!(status.signal(), Some(sigterm));
    /// ```
    ///
    /// A process that receives a signal may catch and handle it, then exit normally with an
    /// exit code. When that happens, `signal` returns `None`.
    ///
    /// Rust does not pass commands through a shell, such as `bash` and `sh`, but it
    /// is possible to do so manually. When invoking a shell, the signal value indicates whether
    /// the top-level shell itself received a terminating signal. If instead a command *within*
    /// an invoked shell receives a terminating signal, many shells convert the signal number
    /// into an exit code by adding 128. For example, a command run under `sh` that receives a
    /// [`SIGTERM`] canonically causes the shell to report an exit code of `15 + 128`, i.e. `143`.
    ///
    /// [`SIGTERM`]: https://pubs.opengroup.org/onlinepubs/9799919799/utilities/kill.html
    /// [`wait`]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/wait.html
    #[stable(feature = "rust1", since = "1.0.0")]
    fn signal(&self) -> Option<i32>;

    /// If the process was terminated by a signal, says whether it dumped core.
    #[stable(feature = "unix_process_wait_more", since = "1.58.0")]
    fn core_dumped(&self) -> bool;

    /// If the process was stopped by a signal, returns that signal.
    ///
    /// In other words, if [`WIFSTOPPED`][`wait`], this returns [`WSTOPSIG`][`wait`].  This is only possible if the status came from
    /// a [`wait`] system call which was passed [`WUNTRACED`][`wait`], and was then converted into an [`ExitStatus`].
    ///
    /// [`wait`]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/wait.html
    #[stable(feature = "unix_process_wait_more", since = "1.58.0")]
    fn stopped_signal(&self) -> Option<i32>;

    /// Whether the process was continued from a stopped status.
    ///
    /// I.e. [`WIFCONTINUED`][`wait`].  This is only possible if the status came from a [`wait`] system call
    /// which was passed [`WCONTINUED`][`wait`], and was then converted into an [`ExitStatus`].
    ///
    /// [`wait`]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/wait.html
    #[stable(feature = "unix_process_wait_more", since = "1.58.0")]
    fn continued(&self) -> bool;

    /// Returns the underlying raw [`wait`] status.
    ///
    /// The returned integer is a **wait status, not an exit status**.
    ///
    /// [`wait`]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/wait.html
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

#[unstable(feature = "unix_send_signal", issue = "141975")]
pub impl(self) trait ChildExt {
    /// Sends a signal to a child process.
    ///
    /// # Errors
    ///
    /// This function will return an error if the signal is invalid. The integer values associated
    /// with signals are implementation-specific, so it's encouraged to use a crate that provides
    /// posix bindings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(unix_send_signal)]
    ///
    /// use std::{io, os::unix::process::ChildExt, process::{Command, Stdio}};
    ///
    /// use libc::SIGTERM;
    ///
    /// fn main() -> io::Result<()> {
    ///     # if cfg!(not(all(target_vendor = "apple", not(target_os = "macos")))) {
    ///     let child = Command::new("cat").stdin(Stdio::piped()).spawn()?;
    ///     child.send_signal(SIGTERM)?;
    ///     # }
    ///     Ok(())
    /// }
    /// ```
    fn send_signal(&self, signal: i32) -> io::Result<()>;

    /// Sends a signal to a child process's process group.
    ///
    /// # Errors
    ///
    /// This function will return an error if the signal is invalid or if the
    /// child process does not have a process group. The integer values
    /// associated with signals are implementation-specific, so it's encouraged
    /// to use a crate that provides posix bindings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(unix_send_signal)]
    ///
    /// use std::{io, os::unix::process::{ChildExt, CommandExt}, process::{Command, Stdio}};
    ///
    /// use libc::SIGTERM;
    ///
    /// fn main() -> io::Result<()> {
    ///     # if cfg!(not(all(target_vendor = "apple", not(target_os = "macos")))) {
    ///     let child = Command::new("cat")
    ///         .stdin(Stdio::piped())
    ///         .process_group(0)
    ///         .spawn()?;
    ///     child.send_process_group_signal(SIGTERM)?;
    ///     # }
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_send_signal", issue = "141975")]
    fn send_process_group_signal(&self, signal: i32) -> io::Result<()>;

    /// Forces the child process's process group to exit.
    ///
    /// This is analogous to [`Child::kill`] but applies to every process in
    /// the child process's process group.
    ///
    /// Use [`CommandExt::process_group`] to assign a child process to an
    /// existing process group, or to make it the leader of a new process group.
    /// By default spawned processes are in the parent's process group.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(unix_kill_process_group)]
    ///
    /// use std::{os::unix::process::{ChildExt, CommandExt}, process::{Command, Stdio}};
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut child = Command::new("cat")
    ///         .stdin(Stdio::piped())
    ///         .process_group(0)
    ///         .spawn()?;
    ///     child.kill_process_group()?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// [`Child::kill`]: process::Child::kill
    #[unstable(feature = "unix_kill_process_group", issue = "156537")]
    fn kill_process_group(&mut self) -> io::Result<()>;
}

#[unstable(feature = "unix_send_signal", issue = "141975")]
impl ChildExt for process::Child {
    fn send_signal(&self, signal: i32) -> io::Result<()> {
        self.handle.send_signal(signal)
    }

    fn send_process_group_signal(&self, signal: i32) -> io::Result<()> {
        self.handle.send_process_group_signal(signal)
    }

    #[cfg(not(target_os = "espidf"))]
    fn kill_process_group(&mut self) -> io::Result<()> {
        self.handle.send_process_group_signal(libc::SIGKILL)
    }

    #[cfg(target_os = "espidf")]
    fn kill_process_group(&mut self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "process groups are not supported on espidf",
        ))
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
        child_stdin.into_inner().into_inner()
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
        let pipe = ChildPipe::from_inner(fd);
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
        child_stdout.into_inner().into_inner()
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
        let pipe = ChildPipe::from_inner(fd);
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
        child_stderr.into_inner().into_inner()
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
        let pipe = ChildPipe::from_inner(fd);
        process::ChildStderr::from_inner(pipe)
    }
}

/// Returns the OS-assigned process identifier associated with this process's parent.
#[must_use]
#[stable(feature = "unix_ppid", since = "1.27.0")]
pub fn parent_id() -> u32 {
    crate::sys::process::getppid()
}
