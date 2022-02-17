//! A module for working with processes.
//!
//! This module is mostly concerned with spawning and interacting with child
//! processes, but it also provides [`abort`] and [`exit`] for terminating the
//! current process.
//!
//! # Spawning a process
//!
//! The [`Command`] struct is used to configure and spawn processes:
//!
//! ```no_run
//! use std::process::Command;
//!
//! let output = Command::new("echo")
//!                      .arg("Hello world")
//!                      .output()
//!                      .expect("Failed to execute command");
//!
//! assert_eq!(b"Hello world\n", output.stdout.as_slice());
//! ```
//!
//! Several methods on [`Command`], such as [`spawn`] or [`output`], can be used
//! to spawn a process. In particular, [`output`] spawns the child process and
//! waits until the process terminates, while [`spawn`] will return a [`Child`]
//! that represents the spawned child process.
//!
//! # Handling I/O
//!
//! The [`stdout`], [`stdin`], and [`stderr`] of a child process can be
//! configured by passing an [`Stdio`] to the corresponding method on
//! [`Command`]. Once spawned, they can be accessed from the [`Child`]. For
//! example, piping output from one command into another command can be done
//! like so:
//!
//! ```no_run
//! use std::process::{Command, Stdio};
//!
//! // stdout must be configured with `Stdio::piped` in order to use
//! // `echo_child.stdout`
//! let echo_child = Command::new("echo")
//!     .arg("Oh no, a tpyo!")
//!     .stdout(Stdio::piped())
//!     .spawn()
//!     .expect("Failed to start echo process");
//!
//! // Note that `echo_child` is moved here, but we won't be needing
//! // `echo_child` anymore
//! let echo_out = echo_child.stdout.expect("Failed to open echo stdout");
//!
//! let mut sed_child = Command::new("sed")
//!     .arg("s/tpyo/typo/")
//!     .stdin(Stdio::from(echo_out))
//!     .stdout(Stdio::piped())
//!     .spawn()
//!     .expect("Failed to start sed process");
//!
//! let output = sed_child.wait_with_output().expect("Failed to wait on sed");
//! assert_eq!(b"Oh no, a typo!\n", output.stdout.as_slice());
//! ```
//!
//! Note that [`ChildStderr`] and [`ChildStdout`] implement [`Read`] and
//! [`ChildStdin`] implements [`Write`]:
//!
//! ```no_run
//! use std::process::{Command, Stdio};
//! use std::io::Write;
//!
//! let mut child = Command::new("/bin/cat")
//!     .stdin(Stdio::piped())
//!     .stdout(Stdio::piped())
//!     .spawn()
//!     .expect("failed to execute child");
//!
//! // If the child process fills its stdout buffer, it may end up
//! // waiting until the parent reads the stdout, and not be able to
//! // read stdin in the meantime, causing a deadlock.
//! // Writing from another thread ensures that stdout is being read
//! // at the same time, avoiding the problem.
//! let mut stdin = child.stdin.take().expect("failed to get stdin");
//! std::thread::spawn(move || {
//!     stdin.write_all(b"test").expect("failed to write to stdin");
//! });
//!
//! let output = child
//!     .wait_with_output()
//!     .expect("failed to wait on child");
//!
//! assert_eq!(b"test", output.stdout.as_slice());
//! ```
//!
//! [`spawn`]: Command::spawn
//! [`output`]: Command::output
//!
//! [`stdout`]: Command::stdout
//! [`stdin`]: Command::stdin
//! [`stderr`]: Command::stderr
//!
//! [`Write`]: io::Write
//! [`Read`]: io::Read

#![stable(feature = "process", since = "1.0.0")]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(test, not(any(target_os = "emscripten", target_env = "sgx"))))]
mod tests;

use crate::io::prelude::*;

use crate::convert::Infallible;
use crate::ffi::OsStr;
use crate::fmt;
use crate::fs;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::num::NonZeroI32;
use crate::path::Path;
use crate::str;
use crate::sys::pipe::{read2, AnonPipe};
use crate::sys::process as imp;
#[stable(feature = "command_access", since = "1.57.0")]
pub use crate::sys_common::process::CommandEnvs;
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};

/// Representation of a running or exited child process.
///
/// This structure is used to represent and manage child processes. A child
/// process is created via the [`Command`] struct, which configures the
/// spawning process and can itself be constructed using a builder-style
/// interface.
///
/// There is no implementation of [`Drop`] for child processes,
/// so if you do not ensure the `Child` has exited then it will continue to
/// run, even after the `Child` handle to the child process has gone out of
/// scope.
///
/// Calling [`wait`] (or other functions that wrap around it) will make
/// the parent process wait until the child has actually exited before
/// continuing.
///
/// # Warning
///
/// On some systems, calling [`wait`] or similar is necessary for the OS to
/// release resources. A process that terminated but has not been waited on is
/// still around as a "zombie". Leaving too many zombies around may exhaust
/// global resources (for example process IDs).
///
/// The standard library does *not* automatically wait on child processes (not
/// even if the `Child` is dropped), it is up to the application developer to do
/// so. As a consequence, dropping `Child` handles without waiting on them first
/// is not recommended in long-running applications.
///
/// # Examples
///
/// ```should_panic
/// use std::process::Command;
///
/// let mut child = Command::new("/bin/cat")
///                         .arg("file.txt")
///                         .spawn()
///                         .expect("failed to execute child");
///
/// let ecode = child.wait()
///                  .expect("failed to wait on child");
///
/// assert!(ecode.success());
/// ```
///
/// [`wait`]: Child::wait
#[stable(feature = "process", since = "1.0.0")]
pub struct Child {
    pub(crate) handle: imp::Process,

    /// The handle for writing to the child's standard input (stdin), if it has
    /// been captured. To avoid partially moving
    /// the `child` and thus blocking yourself from calling
    /// functions on `child` while using `stdin`,
    /// you might find it helpful:
    ///
    /// ```compile_fail,E0425
    /// let stdin = child.stdin.take().unwrap();
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub stdin: Option<ChildStdin>,

    /// The handle for reading from the child's standard output (stdout), if it
    /// has been captured. You might find it helpful to do
    ///
    /// ```compile_fail,E0425
    /// let stdout = child.stdout.take().unwrap();
    /// ```
    ///
    /// to avoid partially moving the `child` and thus blocking yourself from calling
    /// functions on `child` while using `stdout`.
    #[stable(feature = "process", since = "1.0.0")]
    pub stdout: Option<ChildStdout>,

    /// The handle for reading from the child's standard error (stderr), if it
    /// has been captured. You might find it helpful to do
    ///
    /// ```compile_fail,E0425
    /// let stderr = child.stderr.take().unwrap();
    /// ```
    ///
    /// to avoid partially moving the `child` and thus blocking yourself from calling
    /// functions on `child` while using `stderr`.
    #[stable(feature = "process", since = "1.0.0")]
    pub stderr: Option<ChildStderr>,
}

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl crate::sealed::Sealed for Child {}

impl AsInner<imp::Process> for Child {
    fn as_inner(&self) -> &imp::Process {
        &self.handle
    }
}

impl FromInner<(imp::Process, imp::StdioPipes)> for Child {
    fn from_inner((handle, io): (imp::Process, imp::StdioPipes)) -> Child {
        Child {
            handle,
            stdin: io.stdin.map(ChildStdin::from_inner),
            stdout: io.stdout.map(ChildStdout::from_inner),
            stderr: io.stderr.map(ChildStderr::from_inner),
        }
    }
}

impl IntoInner<imp::Process> for Child {
    fn into_inner(self) -> imp::Process {
        self.handle
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Child {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Child")
            .field("stdin", &self.stdin)
            .field("stdout", &self.stdout)
            .field("stderr", &self.stderr)
            .finish_non_exhaustive()
    }
}

/// A handle to a child process's standard input (stdin).
///
/// This struct is used in the [`stdin`] field on [`Child`].
///
/// When an instance of `ChildStdin` is [dropped], the `ChildStdin`'s underlying
/// file handle will be closed. If the child process was blocked on input prior
/// to being dropped, it will become unblocked after dropping.
///
/// [`stdin`]: Child::stdin
/// [dropped]: Drop
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStdin {
    inner: AnonPipe,
}

// In addition to the `impl`s here, `ChildStdin` also has `impl`s for
// `AsFd`/`From<OwnedFd>`/`Into<OwnedFd>` and
// `AsRawFd`/`IntoRawFd`/`FromRawFd`, on Unix and WASI, and
// `AsHandle`/`From<OwnedHandle>`/`Into<OwnedHandle>` and
// `AsRawHandle`/`IntoRawHandle`/`FromRawHandle` on Windows.

#[stable(feature = "process", since = "1.0.0")]
impl Write for ChildStdin {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&*self).write(buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (&*self).write_vectored(bufs)
    }

    fn is_write_vectored(&self) -> bool {
        io::Write::is_write_vectored(&&*self)
    }

    fn flush(&mut self) -> io::Result<()> {
        (&*self).flush()
    }
}

#[stable(feature = "write_mt", since = "1.48.0")]
impl Write for &ChildStdin {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.inner.write_vectored(bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl AsInner<AnonPipe> for ChildStdin {
    fn as_inner(&self) -> &AnonPipe {
        &self.inner
    }
}

impl IntoInner<AnonPipe> for ChildStdin {
    fn into_inner(self) -> AnonPipe {
        self.inner
    }
}

impl FromInner<AnonPipe> for ChildStdin {
    fn from_inner(pipe: AnonPipe) -> ChildStdin {
        ChildStdin { inner: pipe }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for ChildStdin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChildStdin").finish_non_exhaustive()
    }
}

/// A handle to a child process's standard output (stdout).
///
/// This struct is used in the [`stdout`] field on [`Child`].
///
/// When an instance of `ChildStdout` is [dropped], the `ChildStdout`'s
/// underlying file handle will be closed.
///
/// [`stdout`]: Child::stdout
/// [dropped]: Drop
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStdout {
    inner: AnonPipe,
}

// In addition to the `impl`s here, `ChildStdout` also has `impl`s for
// `AsFd`/`From<OwnedFd>`/`Into<OwnedFd>` and
// `AsRawFd`/`IntoRawFd`/`FromRawFd`, on Unix and WASI, and
// `AsHandle`/`From<OwnedHandle>`/`Into<OwnedHandle>` and
// `AsRawHandle`/`IntoRawHandle`/`FromRawHandle` on Windows.

#[stable(feature = "process", since = "1.0.0")]
impl Read for ChildStdout {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.inner.is_read_vectored()
    }
}

impl AsInner<AnonPipe> for ChildStdout {
    fn as_inner(&self) -> &AnonPipe {
        &self.inner
    }
}

impl IntoInner<AnonPipe> for ChildStdout {
    fn into_inner(self) -> AnonPipe {
        self.inner
    }
}

impl FromInner<AnonPipe> for ChildStdout {
    fn from_inner(pipe: AnonPipe) -> ChildStdout {
        ChildStdout { inner: pipe }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for ChildStdout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChildStdout").finish_non_exhaustive()
    }
}

/// A handle to a child process's stderr.
///
/// This struct is used in the [`stderr`] field on [`Child`].
///
/// When an instance of `ChildStderr` is [dropped], the `ChildStderr`'s
/// underlying file handle will be closed.
///
/// [`stderr`]: Child::stderr
/// [dropped]: Drop
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStderr {
    inner: AnonPipe,
}

// In addition to the `impl`s here, `ChildStderr` also has `impl`s for
// `AsFd`/`From<OwnedFd>`/`Into<OwnedFd>` and
// `AsRawFd`/`IntoRawFd`/`FromRawFd`, on Unix and WASI, and
// `AsHandle`/`From<OwnedHandle>`/`Into<OwnedHandle>` and
// `AsRawHandle`/`IntoRawHandle`/`FromRawHandle` on Windows.

#[stable(feature = "process", since = "1.0.0")]
impl Read for ChildStderr {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.inner.read_vectored(bufs)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        self.inner.is_read_vectored()
    }
}

impl AsInner<AnonPipe> for ChildStderr {
    fn as_inner(&self) -> &AnonPipe {
        &self.inner
    }
}

impl IntoInner<AnonPipe> for ChildStderr {
    fn into_inner(self) -> AnonPipe {
        self.inner
    }
}

impl FromInner<AnonPipe> for ChildStderr {
    fn from_inner(pipe: AnonPipe) -> ChildStderr {
        ChildStderr { inner: pipe }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for ChildStderr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChildStderr").finish_non_exhaustive()
    }
}

/// A process builder, providing fine-grained control
/// over how a new process should be spawned.
///
/// A default configuration can be
/// generated using `Command::new(program)`, where `program` gives a path to the
/// program to be executed. Additional builder methods allow the configuration
/// to be changed (for example, by adding arguments) prior to spawning:
///
/// ```
/// use std::process::Command;
///
/// let output = if cfg!(target_os = "windows") {
///     Command::new("cmd")
///             .args(["/C", "echo hello"])
///             .output()
///             .expect("failed to execute process")
/// } else {
///     Command::new("sh")
///             .arg("-c")
///             .arg("echo hello")
///             .output()
///             .expect("failed to execute process")
/// };
///
/// let hello = output.stdout;
/// ```
///
/// `Command` can be reused to spawn multiple processes. The builder methods
/// change the command without needing to immediately spawn the process.
///
/// ```no_run
/// use std::process::Command;
///
/// let mut echo_hello = Command::new("sh");
/// echo_hello.arg("-c")
///           .arg("echo hello");
/// let hello_1 = echo_hello.output().expect("failed to execute process");
/// let hello_2 = echo_hello.output().expect("failed to execute process");
/// ```
///
/// Similarly, you can call builder methods after spawning a process and then
/// spawn a new process with the modified settings.
///
/// ```no_run
/// use std::process::Command;
///
/// let mut list_dir = Command::new("ls");
///
/// // Execute `ls` in the current directory of the program.
/// list_dir.status().expect("process failed to execute");
///
/// println!();
///
/// // Change `ls` to execute in the root directory.
/// list_dir.current_dir("/");
///
/// // And then execute `ls` again but in the root directory.
/// list_dir.status().expect("process failed to execute");
/// ```
#[stable(feature = "process", since = "1.0.0")]
pub struct Command {
    inner: imp::Command,
}

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl crate::sealed::Sealed for Command {}

impl Command {
    /// Constructs a new `Command` for launching the program at
    /// path `program`, with the following default configuration:
    ///
    /// * No arguments to the program
    /// * Inherit the current process's environment
    /// * Inherit the current process's working directory
    /// * Inherit stdin/stdout/stderr for `spawn` or `status`, but create pipes for `output`
    ///
    /// Builder methods are provided to change these defaults and
    /// otherwise configure the process.
    ///
    /// If `program` is not an absolute path, the `PATH` will be searched in
    /// an OS-defined way.
    ///
    /// The search path to be used may be controlled by setting the
    /// `PATH` environment variable on the Command,
    /// but this has some implementation limitations on Windows
    /// (see issue #37519).
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("sh")
    ///         .spawn()
    ///         .expect("sh command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Command {
        Command { inner: imp::Command::new(program.as_ref()) }
    }

    /// Adds an argument to pass to the program.
    ///
    /// Only one argument can be passed per use. So instead of:
    ///
    /// ```no_run
    /// # std::process::Command::new("sh")
    /// .arg("-C /path/to/repo")
    /// # ;
    /// ```
    ///
    /// usage would be:
    ///
    /// ```no_run
    /// # std::process::Command::new("sh")
    /// .arg("-C")
    /// .arg("/path/to/repo")
    /// # ;
    /// ```
    ///
    /// To pass multiple arguments see [`args`].
    ///
    /// [`args`]: Command::args
    ///
    /// Note that the argument is not passed through a shell, but given
    /// literally to the program. This means that shell syntax like quotes,
    /// escaped characters, word splitting, glob patterns, substitution, etc.
    /// have no effect.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .arg("-l")
    ///         .arg("-a")
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Command {
        self.inner.arg(arg.as_ref());
        self
    }

    /// Adds multiple arguments to pass to the program.
    ///
    /// To pass a single argument see [`arg`].
    ///
    /// [`arg`]: Command::arg
    ///
    /// Note that the arguments are not passed through a shell, but given
    /// literally to the program. This means that shell syntax like quotes,
    /// escaped characters, word splitting, glob patterns, substitution, etc.
    /// have no effect.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .args(["-l", "-a"])
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn args<I, S>(&mut self, args: I) -> &mut Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        for arg in args {
            self.arg(arg.as_ref());
        }
        self
    }

    /// Inserts or updates an environment variable mapping.
    ///
    /// Note that environment variable names are case-insensitive (but case-preserving) on Windows,
    /// and case-sensitive on all other platforms.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .env("PATH", "/bin")
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn env<K, V>(&mut self, key: K, val: V) -> &mut Command
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.inner.env_mut().set(key.as_ref(), val.as_ref());
        self
    }

    /// Adds or updates multiple environment variable mappings.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    /// use std::env;
    /// use std::collections::HashMap;
    ///
    /// let filtered_env : HashMap<String, String> =
    ///     env::vars().filter(|&(ref k, _)|
    ///         k == "TERM" || k == "TZ" || k == "LANG" || k == "PATH"
    ///     ).collect();
    ///
    /// Command::new("printenv")
    ///         .stdin(Stdio::null())
    ///         .stdout(Stdio::inherit())
    ///         .env_clear()
    ///         .envs(&filtered_env)
    ///         .spawn()
    ///         .expect("printenv failed to start");
    /// ```
    #[stable(feature = "command_envs", since = "1.19.0")]
    pub fn envs<I, K, V>(&mut self, vars: I) -> &mut Command
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        for (ref key, ref val) in vars {
            self.inner.env_mut().set(key.as_ref(), val.as_ref());
        }
        self
    }

    /// Removes an environment variable mapping.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .env_remove("PATH")
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Command {
        self.inner.env_mut().remove(key.as_ref());
        self
    }

    /// Clears the entire environment map for the child process.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .env_clear()
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn env_clear(&mut self) -> &mut Command {
        self.inner.env_mut().clear();
        self
    }

    /// Sets the working directory for the child process.
    ///
    /// # Platform-specific behavior
    ///
    /// If the program path is relative (e.g., `"./script.sh"`), it's ambiguous
    /// whether it should be interpreted relative to the parent's working
    /// directory or relative to `current_dir`. The behavior in this case is
    /// platform specific and unstable, and it's recommended to use
    /// [`canonicalize`] to get an absolute program path instead.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .current_dir("/bin")
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    ///
    /// [`canonicalize`]: crate::fs::canonicalize
    #[stable(feature = "process", since = "1.0.0")]
    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Command {
        self.inner.cwd(dir.as_ref().as_ref());
        self
    }

    /// Configuration for the child process's standard input (stdin) handle.
    ///
    /// Defaults to [`inherit`] when used with `spawn` or `status`, and
    /// defaults to [`piped`] when used with `output`.
    ///
    /// [`inherit`]: Stdio::inherit
    /// [`piped`]: Stdio::piped
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    ///
    /// Command::new("ls")
    ///         .stdin(Stdio::null())
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stdin<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.inner.stdin(cfg.into().0);
        self
    }

    /// Configuration for the child process's standard output (stdout) handle.
    ///
    /// Defaults to [`inherit`] when used with `spawn` or `status`, and
    /// defaults to [`piped`] when used with `output`.
    ///
    /// [`inherit`]: Stdio::inherit
    /// [`piped`]: Stdio::piped
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    ///
    /// Command::new("ls")
    ///         .stdout(Stdio::null())
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stdout<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.inner.stdout(cfg.into().0);
        self
    }

    /// Configuration for the child process's standard error (stderr) handle.
    ///
    /// Defaults to [`inherit`] when used with `spawn` or `status`, and
    /// defaults to [`piped`] when used with `output`.
    ///
    /// [`inherit`]: Stdio::inherit
    /// [`piped`]: Stdio::piped
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    ///
    /// Command::new("ls")
    ///         .stderr(Stdio::null())
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stderr<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.inner.stderr(cfg.into().0);
        self
    }

    /// Executes the command as a child process, returning a handle to it.
    ///
    /// By default, stdin, stdout and stderr are inherited from the parent.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn spawn(&mut self) -> io::Result<Child> {
        self.inner.spawn(imp::Stdio::Inherit, true).map(Child::from_inner)
    }

    /// Executes the command as a child process, waiting for it to finish and
    /// collecting all of its output.
    ///
    /// By default, stdout and stderr are captured (and used to provide the
    /// resulting output). Stdin is not inherited from the parent and any
    /// attempt by the child process to read from the stdin stream will result
    /// in the stream immediately closing.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::process::Command;
    /// use std::io::{self, Write};
    /// let output = Command::new("/bin/cat")
    ///                      .arg("file.txt")
    ///                      .output()
    ///                      .expect("failed to execute process");
    ///
    /// println!("status: {}", output.status);
    /// io::stdout().write_all(&output.stdout).unwrap();
    /// io::stderr().write_all(&output.stderr).unwrap();
    ///
    /// assert!(output.status.success());
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn output(&mut self) -> io::Result<Output> {
        self.inner
            .spawn(imp::Stdio::MakePipe, false)
            .map(Child::from_inner)
            .and_then(|p| p.wait_with_output())
    }

    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its status.
    ///
    /// By default, stdin, stdout and stderr are inherited from the parent.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::process::Command;
    ///
    /// let status = Command::new("/bin/cat")
    ///                      .arg("file.txt")
    ///                      .status()
    ///                      .expect("failed to execute process");
    ///
    /// println!("process finished with: {}", status);
    ///
    /// assert!(status.success());
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn status(&mut self) -> io::Result<ExitStatus> {
        self.inner
            .spawn(imp::Stdio::Inherit, true)
            .map(Child::from_inner)
            .and_then(|mut p| p.wait())
    }

    /// Returns the path to the program that was given to [`Command::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::process::Command;
    ///
    /// let cmd = Command::new("echo");
    /// assert_eq!(cmd.get_program(), "echo");
    /// ```
    #[must_use]
    #[stable(feature = "command_access", since = "1.57.0")]
    pub fn get_program(&self) -> &OsStr {
        self.inner.get_program()
    }

    /// Returns an iterator of the arguments that will be passed to the program.
    ///
    /// This does not include the path to the program as the first argument;
    /// it only includes the arguments specified with [`Command::arg`] and
    /// [`Command::args`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    /// use std::process::Command;
    ///
    /// let mut cmd = Command::new("echo");
    /// cmd.arg("first").arg("second");
    /// let args: Vec<&OsStr> = cmd.get_args().collect();
    /// assert_eq!(args, &["first", "second"]);
    /// ```
    #[stable(feature = "command_access", since = "1.57.0")]
    pub fn get_args(&self) -> CommandArgs<'_> {
        CommandArgs { inner: self.inner.get_args() }
    }

    /// Returns an iterator of the environment variables that will be set when
    /// the process is spawned.
    ///
    /// Each element is a tuple `(&OsStr, Option<&OsStr>)`, where the first
    /// value is the key, and the second is the value, which is [`None`] if
    /// the environment variable is to be explicitly removed.
    ///
    /// This only includes environment variables explicitly set with
    /// [`Command::env`], [`Command::envs`], and [`Command::env_remove`]. It
    /// does not include environment variables that will be inherited by the
    /// child process.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    /// use std::process::Command;
    ///
    /// let mut cmd = Command::new("ls");
    /// cmd.env("TERM", "dumb").env_remove("TZ");
    /// let envs: Vec<(&OsStr, Option<&OsStr>)> = cmd.get_envs().collect();
    /// assert_eq!(envs, &[
    ///     (OsStr::new("TERM"), Some(OsStr::new("dumb"))),
    ///     (OsStr::new("TZ"), None)
    /// ]);
    /// ```
    #[stable(feature = "command_access", since = "1.57.0")]
    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.inner.get_envs()
    }

    /// Returns the working directory for the child process.
    ///
    /// This returns [`None`] if the working directory will not be changed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use std::process::Command;
    ///
    /// let mut cmd = Command::new("ls");
    /// assert_eq!(cmd.get_current_dir(), None);
    /// cmd.current_dir("/bin");
    /// assert_eq!(cmd.get_current_dir(), Some(Path::new("/bin")));
    /// ```
    #[must_use]
    #[stable(feature = "command_access", since = "1.57.0")]
    pub fn get_current_dir(&self) -> Option<&Path> {
        self.inner.get_current_dir()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Command {
    /// Format the program and arguments of a Command for display. Any
    /// non-utf8 data is lossily converted using the utf8 replacement
    /// character.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl AsInner<imp::Command> for Command {
    fn as_inner(&self) -> &imp::Command {
        &self.inner
    }
}

impl AsInnerMut<imp::Command> for Command {
    fn as_inner_mut(&mut self) -> &mut imp::Command {
        &mut self.inner
    }
}

/// An iterator over the command arguments.
///
/// This struct is created by [`Command::get_args`]. See its documentation for
/// more.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "command_access", since = "1.57.0")]
#[derive(Debug)]
pub struct CommandArgs<'a> {
    inner: imp::CommandArgs<'a>,
}

#[stable(feature = "command_access", since = "1.57.0")]
impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "command_access", since = "1.57.0")]
impl<'a> ExactSizeIterator for CommandArgs<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// The output of a finished process.
///
/// This is returned in a Result by either the [`output`] method of a
/// [`Command`], or the [`wait_with_output`] method of a [`Child`]
/// process.
///
/// [`output`]: Command::output
/// [`wait_with_output`]: Child::wait_with_output
#[derive(PartialEq, Eq, Clone)]
#[stable(feature = "process", since = "1.0.0")]
pub struct Output {
    /// The status (exit code) of the process.
    #[stable(feature = "process", since = "1.0.0")]
    pub status: ExitStatus,
    /// The data that the process wrote to stdout.
    #[stable(feature = "process", since = "1.0.0")]
    pub stdout: Vec<u8>,
    /// The data that the process wrote to stderr.
    #[stable(feature = "process", since = "1.0.0")]
    pub stderr: Vec<u8>,
}

// If either stderr or stdout are valid utf8 strings it prints the valid
// strings, otherwise it prints the byte sequence instead
#[stable(feature = "process_output_debug", since = "1.7.0")]
impl fmt::Debug for Output {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stdout_utf8 = str::from_utf8(&self.stdout);
        let stdout_debug: &dyn fmt::Debug = match stdout_utf8 {
            Ok(ref str) => str,
            Err(_) => &self.stdout,
        };

        let stderr_utf8 = str::from_utf8(&self.stderr);
        let stderr_debug: &dyn fmt::Debug = match stderr_utf8 {
            Ok(ref str) => str,
            Err(_) => &self.stderr,
        };

        fmt.debug_struct("Output")
            .field("status", &self.status)
            .field("stdout", stdout_debug)
            .field("stderr", stderr_debug)
            .finish()
    }
}

/// Describes what to do with a standard I/O stream for a child process when
/// passed to the [`stdin`], [`stdout`], and [`stderr`] methods of [`Command`].
///
/// [`stdin`]: Command::stdin
/// [`stdout`]: Command::stdout
/// [`stderr`]: Command::stderr
#[stable(feature = "process", since = "1.0.0")]
pub struct Stdio(imp::Stdio);

impl Stdio {
    /// A new pipe should be arranged to connect the parent and child processes.
    ///
    /// # Examples
    ///
    /// With stdout:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    ///
    /// let output = Command::new("echo")
    ///     .arg("Hello, world!")
    ///     .stdout(Stdio::piped())
    ///     .output()
    ///     .expect("Failed to execute command");
    ///
    /// assert_eq!(String::from_utf8_lossy(&output.stdout), "Hello, world!\n");
    /// // Nothing echoed to console
    /// ```
    ///
    /// With stdin:
    ///
    /// ```no_run
    /// use std::io::Write;
    /// use std::process::{Command, Stdio};
    ///
    /// let mut child = Command::new("rev")
    ///     .stdin(Stdio::piped())
    ///     .stdout(Stdio::piped())
    ///     .spawn()
    ///     .expect("Failed to spawn child process");
    ///
    /// let mut stdin = child.stdin.take().expect("Failed to open stdin");
    /// std::thread::spawn(move || {
    ///     stdin.write_all("Hello, world!".as_bytes()).expect("Failed to write to stdin");
    /// });
    ///
    /// let output = child.wait_with_output().expect("Failed to read stdout");
    /// assert_eq!(String::from_utf8_lossy(&output.stdout), "!dlrow ,olleH");
    /// ```
    ///
    /// Writing more than a pipe buffer's worth of input to stdin without also reading
    /// stdout and stderr at the same time may cause a deadlock.
    /// This is an issue when running any program that doesn't guarantee that it reads
    /// its entire stdin before writing more than a pipe buffer's worth of output.
    /// The size of a pipe buffer varies on different targets.
    ///
    #[must_use]
    #[stable(feature = "process", since = "1.0.0")]
    pub fn piped() -> Stdio {
        Stdio(imp::Stdio::MakePipe)
    }

    /// The child inherits from the corresponding parent descriptor.
    ///
    /// # Examples
    ///
    /// With stdout:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    ///
    /// let output = Command::new("echo")
    ///     .arg("Hello, world!")
    ///     .stdout(Stdio::inherit())
    ///     .output()
    ///     .expect("Failed to execute command");
    ///
    /// assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    /// // "Hello, world!" echoed to console
    /// ```
    ///
    /// With stdin:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    /// use std::io::{self, Write};
    ///
    /// let output = Command::new("rev")
    ///     .stdin(Stdio::inherit())
    ///     .stdout(Stdio::piped())
    ///     .output()
    ///     .expect("Failed to execute command");
    ///
    /// print!("You piped in the reverse of: ");
    /// io::stdout().write_all(&output.stdout).unwrap();
    /// ```
    #[must_use]
    #[stable(feature = "process", since = "1.0.0")]
    pub fn inherit() -> Stdio {
        Stdio(imp::Stdio::Inherit)
    }

    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`.
    ///
    /// # Examples
    ///
    /// With stdout:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    ///
    /// let output = Command::new("echo")
    ///     .arg("Hello, world!")
    ///     .stdout(Stdio::null())
    ///     .output()
    ///     .expect("Failed to execute command");
    ///
    /// assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    /// // Nothing echoed to console
    /// ```
    ///
    /// With stdin:
    ///
    /// ```no_run
    /// use std::process::{Command, Stdio};
    ///
    /// let output = Command::new("rev")
    ///     .stdin(Stdio::null())
    ///     .stdout(Stdio::piped())
    ///     .output()
    ///     .expect("Failed to execute command");
    ///
    /// assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    /// // Ignores any piped-in input
    /// ```
    #[must_use]
    #[stable(feature = "process", since = "1.0.0")]
    pub fn null() -> Stdio {
        Stdio(imp::Stdio::Null)
    }
}

impl FromInner<imp::Stdio> for Stdio {
    fn from_inner(inner: imp::Stdio) -> Stdio {
        Stdio(inner)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Stdio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stdio").finish_non_exhaustive()
    }
}

#[stable(feature = "stdio_from", since = "1.20.0")]
impl From<ChildStdin> for Stdio {
    /// Converts a [`ChildStdin`] into a [`Stdio`].
    ///
    /// # Examples
    ///
    /// `ChildStdin` will be converted to `Stdio` using `Stdio::from` under the hood.
    ///
    /// ```rust,no_run
    /// use std::process::{Command, Stdio};
    ///
    /// let reverse = Command::new("rev")
    ///     .stdin(Stdio::piped())
    ///     .spawn()
    ///     .expect("failed reverse command");
    ///
    /// let _echo = Command::new("echo")
    ///     .arg("Hello, world!")
    ///     .stdout(reverse.stdin.unwrap()) // Converted into a Stdio here
    ///     .output()
    ///     .expect("failed echo command");
    ///
    /// // "!dlrow ,olleH" echoed to console
    /// ```
    fn from(child: ChildStdin) -> Stdio {
        Stdio::from_inner(child.into_inner().into())
    }
}

#[stable(feature = "stdio_from", since = "1.20.0")]
impl From<ChildStdout> for Stdio {
    /// Converts a [`ChildStdout`] into a [`Stdio`].
    ///
    /// # Examples
    ///
    /// `ChildStdout` will be converted to `Stdio` using `Stdio::from` under the hood.
    ///
    /// ```rust,no_run
    /// use std::process::{Command, Stdio};
    ///
    /// let hello = Command::new("echo")
    ///     .arg("Hello, world!")
    ///     .stdout(Stdio::piped())
    ///     .spawn()
    ///     .expect("failed echo command");
    ///
    /// let reverse = Command::new("rev")
    ///     .stdin(hello.stdout.unwrap())  // Converted into a Stdio here
    ///     .output()
    ///     .expect("failed reverse command");
    ///
    /// assert_eq!(reverse.stdout, b"!dlrow ,olleH\n");
    /// ```
    fn from(child: ChildStdout) -> Stdio {
        Stdio::from_inner(child.into_inner().into())
    }
}

#[stable(feature = "stdio_from", since = "1.20.0")]
impl From<ChildStderr> for Stdio {
    /// Converts a [`ChildStderr`] into a [`Stdio`].
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::process::{Command, Stdio};
    ///
    /// let reverse = Command::new("rev")
    ///     .arg("non_existing_file.txt")
    ///     .stderr(Stdio::piped())
    ///     .spawn()
    ///     .expect("failed reverse command");
    ///
    /// let cat = Command::new("cat")
    ///     .arg("-")
    ///     .stdin(reverse.stderr.unwrap()) // Converted into a Stdio here
    ///     .output()
    ///     .expect("failed echo command");
    ///
    /// assert_eq!(
    ///     String::from_utf8_lossy(&cat.stdout),
    ///     "rev: cannot open non_existing_file.txt: No such file or directory\n"
    /// );
    /// ```
    fn from(child: ChildStderr) -> Stdio {
        Stdio::from_inner(child.into_inner().into())
    }
}

#[stable(feature = "stdio_from", since = "1.20.0")]
impl From<fs::File> for Stdio {
    /// Converts a [`File`](fs::File) into a [`Stdio`].
    ///
    /// # Examples
    ///
    /// `File` will be converted to `Stdio` using `Stdio::from` under the hood.
    ///
    /// ```rust,no_run
    /// use std::fs::File;
    /// use std::process::Command;
    ///
    /// // With the `foo.txt` file containing `Hello, world!"
    /// let file = File::open("foo.txt").unwrap();
    ///
    /// let reverse = Command::new("rev")
    ///     .stdin(file)  // Implicit File conversion into a Stdio
    ///     .output()
    ///     .expect("failed reverse command");
    ///
    /// assert_eq!(reverse.stdout, b"!dlrow ,olleH");
    /// ```
    fn from(file: fs::File) -> Stdio {
        Stdio::from_inner(file.into_inner().into())
    }
}

/// Describes the result of a process after it has terminated.
///
/// This `struct` is used to represent the exit status or other termination of a child process.
/// Child processes are created via the [`Command`] struct and their exit
/// status is exposed through the [`status`] method, or the [`wait`] method
/// of a [`Child`] process.
///
/// An `ExitStatus` represents every possible disposition of a process.  On Unix this
/// is the **wait status**.  It is *not* simply an *exit status* (a value passed to `exit`).
///
/// For proper error reporting of failed processes, print the value of `ExitStatus` or
/// `ExitStatusError` using their implementations of [`Display`](crate::fmt::Display).
///
/// [`status`]: Command::status
/// [`wait`]: Child::wait
//
// We speak slightly loosely (here and in various other places in the stdlib docs) about `exit`
// vs `_exit`.  Naming of Unix system calls is not standardised across Unices, so terminology is a
// matter of convention and tradition.  For clarity we usually speak of `exit`, even when we might
// mean an underlying system call such as `_exit`.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "process", since = "1.0.0")]
pub struct ExitStatus(imp::ExitStatus);

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl crate::sealed::Sealed for ExitStatus {}

impl ExitStatus {
    /// Was termination successful?  Returns a `Result`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(exit_status_error)]
    /// # if cfg!(unix) {
    /// use std::process::Command;
    ///
    /// let status = Command::new("ls")
    ///                      .arg("/dev/nonexistent")
    ///                      .status()
    ///                      .expect("ls could not be executed");
    ///
    /// println!("ls: {}", status);
    /// status.exit_ok().expect_err("/dev/nonexistent could be listed!");
    /// # } // cfg!(unix)
    /// ```
    #[unstable(feature = "exit_status_error", issue = "84908")]
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        self.0.exit_ok().map_err(ExitStatusError)
    }

    /// Was termination successful? Signal termination is not considered a
    /// success, and success is defined as a zero exit status.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::process::Command;
    ///
    /// let status = Command::new("mkdir")
    ///                      .arg("projects")
    ///                      .status()
    ///                      .expect("failed to execute mkdir");
    ///
    /// if status.success() {
    ///     println!("'projects/' directory created");
    /// } else {
    ///     println!("failed to create 'projects/' directory: {}", status);
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "process", since = "1.0.0")]
    pub fn success(&self) -> bool {
        self.0.exit_ok().is_ok()
    }

    /// Returns the exit code of the process, if any.
    ///
    /// In Unix terms the return value is the **exit status**: the value passed to `exit`, if the
    /// process finished by calling `exit`.  Note that on Unix the exit status is truncated to 8
    /// bits, and that values that didn't come from a program's call to `exit` may be invented by the
    /// runtime system (often, for example, 255, 254, 127 or 126).
    ///
    /// On Unix, this will return `None` if the process was terminated by a signal.
    /// [`ExitStatusExt`](crate::os::unix::process::ExitStatusExt) is an
    /// extension trait for extracting any such signal, and other details, from the `ExitStatus`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// let status = Command::new("mkdir")
    ///                      .arg("projects")
    ///                      .status()
    ///                      .expect("failed to execute mkdir");
    ///
    /// match status.code() {
    ///     Some(code) => println!("Exited with status code: {}", code),
    ///     None       => println!("Process terminated by signal")
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "process", since = "1.0.0")]
    pub fn code(&self) -> Option<i32> {
        self.0.code()
    }
}

impl AsInner<imp::ExitStatus> for ExitStatus {
    fn as_inner(&self) -> &imp::ExitStatus {
        &self.0
    }
}

impl FromInner<imp::ExitStatus> for ExitStatus {
    fn from_inner(s: imp::ExitStatus) -> ExitStatus {
        ExitStatus(s)
    }
}

#[stable(feature = "process", since = "1.0.0")]
impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Allows extension traits within `std`.
#[unstable(feature = "sealed", issue = "none")]
impl crate::sealed::Sealed for ExitStatusError {}

/// Describes the result of a process after it has failed
///
/// Produced by the [`.exit_ok`](ExitStatus::exit_ok) method on [`ExitStatus`].
///
/// # Examples
///
/// ```
/// #![feature(exit_status_error)]
/// # if cfg!(unix) {
/// use std::process::{Command, ExitStatusError};
///
/// fn run(cmd: &str) -> Result<(),ExitStatusError> {
///     Command::new(cmd).status().unwrap().exit_ok()?;
///     Ok(())
/// }
///
/// run("true").unwrap();
/// run("false").unwrap_err();
/// # } // cfg!(unix)
/// ```
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[unstable(feature = "exit_status_error", issue = "84908")]
// The definition of imp::ExitStatusError should ideally be such that
// Result<(), imp::ExitStatusError> has an identical representation to imp::ExitStatus.
pub struct ExitStatusError(imp::ExitStatusError);

#[unstable(feature = "exit_status_error", issue = "84908")]
impl ExitStatusError {
    /// Reports the exit code, if applicable, from an `ExitStatusError`.
    ///
    /// In Unix terms the return value is the **exit status**: the value passed to `exit`, if the
    /// process finished by calling `exit`.  Note that on Unix the exit status is truncated to 8
    /// bits, and that values that didn't come from a program's call to `exit` may be invented by the
    /// runtime system (often, for example, 255, 254, 127 or 126).
    ///
    /// On Unix, this will return `None` if the process was terminated by a signal.  If you want to
    /// handle such situations specially, consider using methods from
    /// [`ExitStatusExt`](crate::os::unix::process::ExitStatusExt).
    ///
    /// If the process finished by calling `exit` with a nonzero value, this will return
    /// that exit status.
    ///
    /// If the error was something else, it will return `None`.
    ///
    /// If the process exited successfully (ie, by calling `exit(0)`), there is no
    /// `ExitStatusError`.  So the return value from `ExitStatusError::code()` is always nonzero.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(exit_status_error)]
    /// # #[cfg(unix)] {
    /// use std::process::Command;
    ///
    /// let bad = Command::new("false").status().unwrap().exit_ok().unwrap_err();
    /// assert_eq!(bad.code(), Some(1));
    /// # } // #[cfg(unix)]
    /// ```
    #[must_use]
    pub fn code(&self) -> Option<i32> {
        self.code_nonzero().map(Into::into)
    }

    /// Reports the exit code, if applicable, from an `ExitStatusError`, as a `NonZero`
    ///
    /// This is exactly like [`code()`](Self::code), except that it returns a `NonZeroI32`.
    ///
    /// Plain `code`, returning a plain integer, is provided because is is often more convenient.
    /// The returned value from `code()` is indeed also nonzero; use `code_nonzero()` when you want
    /// a type-level guarantee of nonzeroness.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(exit_status_error)]
    /// # if cfg!(unix) {
    /// use std::num::NonZeroI32;
    /// use std::process::Command;
    ///
    /// let bad = Command::new("false").status().unwrap().exit_ok().unwrap_err();
    /// assert_eq!(bad.code_nonzero().unwrap(), NonZeroI32::try_from(1).unwrap());
    /// # } // cfg!(unix)
    /// ```
    #[must_use]
    pub fn code_nonzero(&self) -> Option<NonZeroI32> {
        self.0.code()
    }

    /// Converts an `ExitStatusError` (back) to an `ExitStatus`.
    #[must_use]
    pub fn into_status(&self) -> ExitStatus {
        ExitStatus(self.0.into())
    }
}

#[unstable(feature = "exit_status_error", issue = "84908")]
impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0.into())
    }
}

#[unstable(feature = "exit_status_error", issue = "84908")]
impl fmt::Display for ExitStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "process exited unsuccessfully: {}", self.into_status())
    }
}

#[unstable(feature = "exit_status_error", issue = "84908")]
impl crate::error::Error for ExitStatusError {}

/// This type represents the status code a process can return to its
/// parent under normal termination.
///
/// Numeric values used in this type don't have portable meanings, and
/// different platforms may mask different amounts of them.
///
/// For the platform's canonical successful and unsuccessful codes, see
/// the [`SUCCESS`] and [`FAILURE`] associated items.
///
/// [`SUCCESS`]: ExitCode::SUCCESS
/// [`FAILURE`]: ExitCode::FAILURE
///
/// **Warning**: While various forms of this were discussed in [RFC #1937],
/// it was ultimately cut from that RFC, and thus this type is more subject
/// to change even than the usual unstable item churn.
///
/// [RFC #1937]: https://github.com/rust-lang/rfcs/pull/1937
#[derive(Clone, Copy, Debug)]
#[unstable(feature = "process_exitcode_placeholder", issue = "48711")]
pub struct ExitCode(imp::ExitCode);

#[unstable(feature = "process_exitcode_placeholder", issue = "48711")]
impl ExitCode {
    /// The canonical ExitCode for successful termination on this platform.
    ///
    /// Note that a `()`-returning `main` implicitly results in a successful
    /// termination, so there's no need to return this from `main` unless
    /// you're also returning other possible codes.
    #[unstable(feature = "process_exitcode_placeholder", issue = "48711")]
    pub const SUCCESS: ExitCode = ExitCode(imp::ExitCode::SUCCESS);

    /// The canonical ExitCode for unsuccessful termination on this platform.
    ///
    /// If you're only returning this and `SUCCESS` from `main`, consider
    /// instead returning `Err(_)` and `Ok(())` respectively, which will
    /// return the same codes (but will also `eprintln!` the error).
    #[unstable(feature = "process_exitcode_placeholder", issue = "48711")]
    pub const FAILURE: ExitCode = ExitCode(imp::ExitCode::FAILURE);
}

impl ExitCode {
    // This should not be stabilized when stabilizing ExitCode, we don't know that i32 will serve
    // all usecases, for example windows seems to use u32, unix uses the 8-15th bits of an i32, we
    // likely want to isolate users anything that could restrict the platform specific
    // representation of an ExitCode
    //
    // More info: https://internals.rust-lang.org/t/mini-pre-rfc-redesigning-process-exitstatus/5426
    /// Convert an ExitCode into an i32
    #[unstable(feature = "process_exitcode_placeholder", issue = "48711")]
    #[inline]
    pub fn to_i32(self) -> i32 {
        self.0.as_i32()
    }
}

#[unstable(feature = "process_exitcode_placeholder", issue = "48711")]
impl From<u8> for ExitCode {
    /// Construct an exit code from an arbitrary u8 value.
    fn from(code: u8) -> Self {
        ExitCode(imp::ExitCode::from(code))
    }
}

impl Child {
    /// Forces the child process to exit. If the child has already exited, an [`InvalidInput`]
    /// error is returned.
    ///
    /// The mapping to [`ErrorKind`]s is not part of the compatibility contract of the function.
    ///
    /// This is equivalent to sending a SIGKILL on Unix platforms.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// let mut command = Command::new("yes");
    /// if let Ok(mut child) = command.spawn() {
    ///     child.kill().expect("command wasn't running");
    /// } else {
    ///     println!("yes command didn't start");
    /// }
    /// ```
    ///
    /// [`ErrorKind`]: io::ErrorKind
    /// [`InvalidInput`]: io::ErrorKind::InvalidInput
    #[stable(feature = "process", since = "1.0.0")]
    pub fn kill(&mut self) -> io::Result<()> {
        self.handle.kill()
    }

    /// Returns the OS-assigned process identifier associated with this child.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// let mut command = Command::new("ls");
    /// if let Ok(child) = command.spawn() {
    ///     println!("Child's ID is {}", child.id());
    /// } else {
    ///     println!("ls command didn't start");
    /// }
    /// ```
    #[must_use]
    #[stable(feature = "process_id", since = "1.3.0")]
    pub fn id(&self) -> u32 {
        self.handle.id()
    }

    /// Waits for the child to exit completely, returning the status that it
    /// exited with. This function will continue to have the same return value
    /// after it has been called at least once.
    ///
    /// The stdin handle to the child process, if any, will be closed
    /// before waiting. This helps avoid deadlock: it ensures that the
    /// child does not block waiting for input from the parent, while
    /// the parent waits for the child to exit.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// let mut command = Command::new("ls");
    /// if let Ok(mut child) = command.spawn() {
    ///     child.wait().expect("command wasn't running");
    ///     println!("Child has finished its execution!");
    /// } else {
    ///     println!("ls command didn't start");
    /// }
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        drop(self.stdin.take());
        self.handle.wait().map(ExitStatus)
    }

    /// Attempts to collect the exit status of the child if it has already
    /// exited.
    ///
    /// This function will not block the calling thread and will only
    /// check to see if the child process has exited or not. If the child has
    /// exited then on Unix the process ID is reaped. This function is
    /// guaranteed to repeatedly return a successful exit status so long as the
    /// child has already exited.
    ///
    /// If the child has exited, then `Ok(Some(status))` is returned. If the
    /// exit status is not available at this time then `Ok(None)` is returned.
    /// If an error occurs, then that error is returned.
    ///
    /// Note that unlike `wait`, this function will not attempt to drop stdin.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// let mut child = Command::new("ls").spawn().unwrap();
    ///
    /// match child.try_wait() {
    ///     Ok(Some(status)) => println!("exited with: {}", status),
    ///     Ok(None) => {
    ///         println!("status not ready yet, let's really wait");
    ///         let res = child.wait();
    ///         println!("result: {:?}", res);
    ///     }
    ///     Err(e) => println!("error attempting to wait: {}", e),
    /// }
    /// ```
    #[stable(feature = "process_try_wait", since = "1.18.0")]
    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        Ok(self.handle.try_wait()?.map(ExitStatus))
    }

    /// Simultaneously waits for the child to exit and collect all remaining
    /// output on the stdout/stderr handles, returning an `Output`
    /// instance.
    ///
    /// The stdin handle to the child process, if any, will be closed
    /// before waiting. This helps avoid deadlock: it ensures that the
    /// child does not block waiting for input from the parent, while
    /// the parent waits for the child to exit.
    ///
    /// By default, stdin, stdout and stderr are inherited from the parent.
    /// In order to capture the output into this `Result<Output>` it is
    /// necessary to create new pipes between parent and child. Use
    /// `stdout(Stdio::piped())` or `stderr(Stdio::piped())`, respectively.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::process::{Command, Stdio};
    ///
    /// let child = Command::new("/bin/cat")
    ///     .arg("file.txt")
    ///     .stdout(Stdio::piped())
    ///     .spawn()
    ///     .expect("failed to execute child");
    ///
    /// let output = child
    ///     .wait_with_output()
    ///     .expect("failed to wait on child");
    ///
    /// assert!(output.status.success());
    /// ```
    ///
    #[stable(feature = "process", since = "1.0.0")]
    pub fn wait_with_output(mut self) -> io::Result<Output> {
        drop(self.stdin.take());

        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        match (self.stdout.take(), self.stderr.take()) {
            (None, None) => {}
            (Some(mut out), None) => {
                let res = out.read_to_end(&mut stdout);
                res.unwrap();
            }
            (None, Some(mut err)) => {
                let res = err.read_to_end(&mut stderr);
                res.unwrap();
            }
            (Some(out), Some(err)) => {
                let res = read2(out.inner, &mut stdout, err.inner, &mut stderr);
                res.unwrap();
            }
        }

        let status = self.wait()?;
        Ok(Output { status, stdout, stderr })
    }
}

/// Terminates the current process with the specified exit code.
///
/// This function will never return and will immediately terminate the current
/// process. The exit code is passed through to the underlying OS and will be
/// available for consumption by another process.
///
/// Note that because this function never returns, and that it terminates the
/// process, no destructors on the current stack or any other thread's stack
/// will be run. If a clean shutdown is needed it is recommended to only call
/// this function at a known point where there are no more destructors left
/// to run.
///
/// ## Platform-specific behavior
///
/// **Unix**: On Unix-like platforms, it is unlikely that all 32 bits of `exit`
/// will be visible to a parent process inspecting the exit code. On most
/// Unix-like platforms, only the eight least-significant bits are considered.
///
/// # Examples
///
/// Due to this function’s behavior regarding destructors, a conventional way
/// to use the function is to extract the actual computation to another
/// function and compute the exit code from its return value:
///
/// ```
/// fn run_app() -> Result<(), ()> {
///     // Application logic here
///     Ok(())
/// }
///
/// fn main() {
///     std::process::exit(match run_app() {
///         Ok(_) => 0,
///         Err(err) => {
///             eprintln!("error: {:?}", err);
///             1
///         }
///     });
/// }
/// ```
///
/// Due to [platform-specific behavior], the exit code for this example will be
/// `0` on Linux, but `256` on Windows:
///
/// ```no_run
/// use std::process;
///
/// process::exit(0x0100);
/// ```
///
/// [platform-specific behavior]: #platform-specific-behavior
#[stable(feature = "rust1", since = "1.0.0")]
pub fn exit(code: i32) -> ! {
    crate::rt::cleanup();
    crate::sys::os::exit(code)
}

/// Terminates the process in an abnormal fashion.
///
/// The function will never return and will immediately terminate the current
/// process in a platform specific "abnormal" manner.
///
/// Note that because this function never returns, and that it terminates the
/// process, no destructors on the current stack or any other thread's stack
/// will be run.
///
/// Rust IO buffers (eg, from `BufWriter`) will not be flushed.
/// Likewise, C stdio buffers will (on most platforms) not be flushed.
///
/// This is in contrast to the default behaviour of [`panic!`] which unwinds
/// the current thread's stack and calls all destructors.
/// When `panic="abort"` is set, either as an argument to `rustc` or in a
/// crate's Cargo.toml, [`panic!`] and `abort` are similar. However,
/// [`panic!`] will still call the [panic hook] while `abort` will not.
///
/// If a clean shutdown is needed it is recommended to only call
/// this function at a known point where there are no more destructors left
/// to run.
///
/// The process's termination will be similar to that from the C `abort()`
/// function.  On Unix, the process will terminate with signal `SIGABRT`, which
/// typically means that the shell prints "Aborted".
///
/// # Examples
///
/// ```no_run
/// use std::process;
///
/// fn main() {
///     println!("aborting");
///
///     process::abort();
///
///     // execution never gets here
/// }
/// ```
///
/// The `abort` function terminates the process, so the destructor will not
/// get run on the example below:
///
/// ```no_run
/// use std::process;
///
/// struct HasDrop;
///
/// impl Drop for HasDrop {
///     fn drop(&mut self) {
///         println!("This will never be printed!");
///     }
/// }
///
/// fn main() {
///     let _x = HasDrop;
///     process::abort();
///     // the destructor implemented for HasDrop will never get run
/// }
/// ```
///
/// [panic hook]: crate::panic::set_hook
#[stable(feature = "process_abort", since = "1.17.0")]
#[cold]
pub fn abort() -> ! {
    crate::sys::abort_internal();
}

/// Returns the OS-assigned process identifier associated with this process.
///
/// # Examples
///
/// Basic usage:
///
/// ```no_run
/// use std::process;
///
/// println!("My pid is {}", process::id());
/// ```
///
///
#[must_use]
#[stable(feature = "getpid", since = "1.26.0")]
pub fn id() -> u32 {
    crate::sys::os::getpid()
}

/// A trait for implementing arbitrary return types in the `main` function.
///
/// The C-main function only supports to return integers as return type.
/// So, every type implementing the `Termination` trait has to be converted
/// to an integer.
///
/// The default implementations are returning `libc::EXIT_SUCCESS` to indicate
/// a successful execution. In case of a failure, `libc::EXIT_FAILURE` is returned.
#[cfg_attr(not(test), lang = "termination")]
#[unstable(feature = "termination_trait_lib", issue = "43301")]
#[rustc_on_unimplemented(
    message = "`main` has invalid return type `{Self}`",
    label = "`main` can only return types that implement `{Termination}`"
)]
pub trait Termination {
    /// Is called to get the representation of the value as status code.
    /// This status code is returned to the operating system.
    fn report(self) -> ExitCode;
}

#[unstable(feature = "termination_trait_lib", issue = "43301")]
impl Termination for () {
    #[inline]
    fn report(self) -> ExitCode {
        ExitCode::SUCCESS.report()
    }
}

#[unstable(feature = "termination_trait_lib", issue = "43301")]
impl<E: fmt::Debug> Termination for Result<(), E> {
    fn report(self) -> ExitCode {
        match self {
            Ok(()) => ().report(),
            Err(err) => Err::<!, _>(err).report(),
        }
    }
}

#[unstable(feature = "termination_trait_lib", issue = "43301")]
impl Termination for ! {
    fn report(self) -> ExitCode {
        self
    }
}

#[unstable(feature = "termination_trait_lib", issue = "43301")]
impl<E: fmt::Debug> Termination for Result<!, E> {
    fn report(self) -> ExitCode {
        let Err(err) = self;
        eprintln!("Error: {:?}", err);
        ExitCode::FAILURE.report()
    }
}

#[unstable(feature = "termination_trait_lib", issue = "43301")]
impl<E: fmt::Debug> Termination for Result<Infallible, E> {
    fn report(self) -> ExitCode {
        let Err(err) = self;
        Err::<!, _>(err).report()
    }
}

#[unstable(feature = "termination_trait_lib", issue = "43301")]
impl Termination for ExitCode {
    #[inline]
    fn report(self) -> ExitCode {
        self
    }
}
