// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A module for working with processes.
//!
//! # Examples
//!
//! Basic usage where we try to execute the `cat` shell command:
//!
//! ```should_panic
//! use std::process::Command;
//!
//! let mut child = Command::new("/bin/cat")
//!                         .arg("file.txt")
//!                         .spawn()
//!                         .expect("failed to execute child");
//!
//! let ecode = child.wait()
//!                  .expect("failed to wait on child");
//!
//! assert!(ecode.success());
//! ```

#![stable(feature = "process", since = "1.0.0")]

use io::prelude::*;

use ffi::OsStr;
use fmt;
use io;
use path::Path;
use str;
use sys::pipe::{read2, AnonPipe};
use sys::process as imp;
use sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};

/// Representation of a running or exited child process.
///
/// This structure is used to represent and manage child processes. A child
/// process is created via the [`Command`] struct, which configures the
/// spawning process and can itself be constructed using a builder-style
/// interface.
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
/// # Note
///
/// Take note that there is no implementation of [`Drop`] for child processes,
/// so if you do not ensure the `Child` has exited then it will continue to
/// run, even after the `Child` handle to the child process has gone out of
/// scope.
///
/// Calling [`wait`][`wait`] (or other functions that wrap around it) will make
/// the parent process wait until the child has actually exited before
/// continuing.
///
/// [`Command`]: struct.Command.html
/// [`Drop`]: ../../core/ops/trait.Drop.html
/// [`wait`]: #method.wait
#[stable(feature = "process", since = "1.0.0")]
pub struct Child {
    handle: imp::Process,

    /// The handle for writing to the child's stdin, if it has been captured
    #[stable(feature = "process", since = "1.0.0")]
    pub stdin: Option<ChildStdin>,

    /// The handle for reading from the child's stdout, if it has been captured
    #[stable(feature = "process", since = "1.0.0")]
    pub stdout: Option<ChildStdout>,

    /// The handle for reading from the child's stderr, if it has been captured
    #[stable(feature = "process", since = "1.0.0")]
    pub stderr: Option<ChildStderr>,
}

impl AsInner<imp::Process> for Child {
    fn as_inner(&self) -> &imp::Process { &self.handle }
}

impl FromInner<(imp::Process, imp::StdioPipes)> for Child {
    fn from_inner((handle, io): (imp::Process, imp::StdioPipes)) -> Child {
        Child {
            handle: handle,
            stdin: io.stdin.map(ChildStdin::from_inner),
            stdout: io.stdout.map(ChildStdout::from_inner),
            stderr: io.stderr.map(ChildStderr::from_inner),
        }
    }
}

impl IntoInner<imp::Process> for Child {
    fn into_inner(self) -> imp::Process { self.handle }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for Child {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Child")
            .field("stdin", &self.stdin)
            .field("stdout", &self.stdout)
            .field("stderr", &self.stderr)
            .finish()
    }
}

/// A handle to a child process's stdin. This struct is used in the [`stdin`]
/// field on [`Child`].
///
/// [`Child`]: struct.Child.html
/// [`stdin`]: struct.Child.html#structfield.stdin
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStdin {
    inner: AnonPipe
}

#[stable(feature = "process", since = "1.0.0")]
impl Write for ChildStdin {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl AsInner<AnonPipe> for ChildStdin {
    fn as_inner(&self) -> &AnonPipe { &self.inner }
}

impl IntoInner<AnonPipe> for ChildStdin {
    fn into_inner(self) -> AnonPipe { self.inner }
}

impl FromInner<AnonPipe> for ChildStdin {
    fn from_inner(pipe: AnonPipe) -> ChildStdin {
        ChildStdin { inner: pipe }
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for ChildStdin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("ChildStdin { .. }")
    }
}

/// A handle to a child process's stdout. This struct is used in the [`stdout`]
/// field on [`Child`].
///
/// [`Child`]: struct.Child.html
/// [`stdout`]: struct.Child.html#structfield.stdout
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStdout {
    inner: AnonPipe
}

#[stable(feature = "process", since = "1.0.0")]
impl Read for ChildStdout {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.inner.read_to_end(buf)
    }
}

impl AsInner<AnonPipe> for ChildStdout {
    fn as_inner(&self) -> &AnonPipe { &self.inner }
}

impl IntoInner<AnonPipe> for ChildStdout {
    fn into_inner(self) -> AnonPipe { self.inner }
}

impl FromInner<AnonPipe> for ChildStdout {
    fn from_inner(pipe: AnonPipe) -> ChildStdout {
        ChildStdout { inner: pipe }
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for ChildStdout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("ChildStdout { .. }")
    }
}

/// A handle to a child process's stderr. This struct is used in the [`stderr`]
/// field on [`Child`].
///
/// [`Child`]: struct.Child.html
/// [`stderr`]: struct.Child.html#structfield.stderr
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStderr {
    inner: AnonPipe
}

#[stable(feature = "process", since = "1.0.0")]
impl Read for ChildStderr {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.inner.read_to_end(buf)
    }
}

impl AsInner<AnonPipe> for ChildStderr {
    fn as_inner(&self) -> &AnonPipe { &self.inner }
}

impl IntoInner<AnonPipe> for ChildStderr {
    fn into_inner(self) -> AnonPipe { self.inner }
}

impl FromInner<AnonPipe> for ChildStderr {
    fn from_inner(pipe: AnonPipe) -> ChildStderr {
        ChildStderr { inner: pipe }
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for ChildStderr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("ChildStderr { .. }")
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
///             .args(&["/C", "echo hello"])
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
#[stable(feature = "process", since = "1.0.0")]
pub struct Command {
    inner: imp::Command,
}

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
    /// (see https://github.com/rust-lang/rust/issues/37519).
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

    /// Add an argument to pass to the program.
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

    /// Add multiple arguments to pass to the program.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::process::Command;
    ///
    /// Command::new("ls")
    ///         .args(&["-l", "-a"])
    ///         .spawn()
    ///         .expect("ls command failed to start");
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn args<S: AsRef<OsStr>>(&mut self, args: &[S]) -> &mut Command {
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
        where K: AsRef<OsStr>, V: AsRef<OsStr>
    {
        self.inner.env(key.as_ref(), val.as_ref());
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
        self.inner.env_remove(key.as_ref());
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
        self.inner.env_clear();
        self
    }

    /// Sets the working directory for the child process.
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
    #[stable(feature = "process", since = "1.0.0")]
    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Command {
        self.inner.cwd(dir.as_ref().as_ref());
        self
    }

    /// Configuration for the child process's stdin handle (file descriptor 0).
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
    pub fn stdin(&mut self, cfg: Stdio) -> &mut Command {
        self.inner.stdin(cfg.0);
        self
    }

    /// Configuration for the child process's stdout handle (file descriptor 1).
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
    pub fn stdout(&mut self, cfg: Stdio) -> &mut Command {
        self.inner.stdout(cfg.0);
        self
    }

    /// Configuration for the child process's stderr handle (file descriptor 2).
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
    pub fn stderr(&mut self, cfg: Stdio) -> &mut Command {
        self.inner.stderr(cfg.0);
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
    /// By default, stdin, stdout and stderr are captured (and used to
    /// provide the resulting output).
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::process::Command;
    /// let output = Command::new("/bin/cat")
    ///                      .arg("file.txt")
    ///                      .output()
    ///                      .expect("failed to execute process");
    ///
    /// println!("status: {}", output.status);
    /// println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    /// println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    ///
    /// assert!(output.status.success());
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn output(&mut self) -> io::Result<Output> {
        self.inner.spawn(imp::Stdio::MakePipe, false).map(Child::from_inner)
            .and_then(|p| p.wait_with_output())
    }

    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its exit status.
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
    /// println!("process exited with: {}", status);
    ///
    /// assert!(status.success());
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn status(&mut self) -> io::Result<ExitStatus> {
        self.inner.spawn(imp::Stdio::Inherit, true).map(Child::from_inner)
                  .and_then(|mut p| p.wait())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Command {
    /// Format the program and arguments of a Command for display. Any
    /// non-utf8 data is lossily converted using the utf8 replacement
    /// character.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl AsInner<imp::Command> for Command {
    fn as_inner(&self) -> &imp::Command { &self.inner }
}

impl AsInnerMut<imp::Command> for Command {
    fn as_inner_mut(&mut self) -> &mut imp::Command { &mut self.inner }
}

/// The output of a finished process.
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
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {

        let stdout_utf8 = str::from_utf8(&self.stdout);
        let stdout_debug: &fmt::Debug = match stdout_utf8 {
            Ok(ref str) => str,
            Err(_) => &self.stdout
        };

        let stderr_utf8 = str::from_utf8(&self.stderr);
        let stderr_debug: &fmt::Debug = match stderr_utf8 {
            Ok(ref str) => str,
            Err(_) => &self.stderr
        };

        fmt.debug_struct("Output")
            .field("status", &self.status)
            .field("stdout", stdout_debug)
            .field("stderr", stderr_debug)
            .finish()
    }
}

/// Describes what to do with a standard I/O stream for a child process.
#[stable(feature = "process", since = "1.0.0")]
pub struct Stdio(imp::Stdio);

impl Stdio {
    /// A new pipe should be arranged to connect the parent and child processes.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn piped() -> Stdio { Stdio(imp::Stdio::MakePipe) }

    /// The child inherits from the corresponding parent descriptor.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn inherit() -> Stdio { Stdio(imp::Stdio::Inherit) }

    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    #[stable(feature = "process", since = "1.0.0")]
    pub fn null() -> Stdio { Stdio(imp::Stdio::Null) }
}

impl FromInner<imp::Stdio> for Stdio {
    fn from_inner(inner: imp::Stdio) -> Stdio {
        Stdio(inner)
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl fmt::Debug for Stdio {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Stdio { .. }")
    }
}

/// Describes the result of a process after it has terminated.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "process", since = "1.0.0")]
pub struct ExitStatus(imp::ExitStatus);

impl ExitStatus {
    /// Was termination successful? Signal termination not considered a success,
    /// and success is defined as a zero exit status.
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
    ///     println!("failed to create 'projects/' directory");
    /// }
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn success(&self) -> bool {
        self.0.success()
    }

    /// Returns the exit code of the process, if any.
    ///
    /// On Unix, this will return `None` if the process was terminated
    /// by a signal; `std::os::unix` provides an extension trait for
    /// extracting the signal and other details from the `ExitStatus`.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn code(&self) -> Option<i32> {
        self.0.code()
    }
}

impl AsInner<imp::ExitStatus> for ExitStatus {
    fn as_inner(&self) -> &imp::ExitStatus { &self.0 }
}

impl FromInner<imp::ExitStatus> for ExitStatus {
    fn from_inner(s: imp::ExitStatus) -> ExitStatus {
        ExitStatus(s)
    }
}

#[stable(feature = "process", since = "1.0.0")]
impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Child {
    /// Forces the child to exit. This is equivalent to sending a
    /// SIGKILL on unix platforms.
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
    ///     println!("Child's id is {}", child.id());
    /// } else {
    ///     println!("ls command didn't start");
    /// }
    /// ```
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
    /// This function will not block the calling thread and will only advisorily
    /// check to see if the child process has exited or not. If the child has
    /// exited then on Unix the process id is reaped. This function is
    /// guaranteed to repeatedly return a successful exit status so long as the
    /// child has already exited.
    ///
    /// If the child has exited, then `Ok(status)` is returned. If the exit
    /// status is not available at this time then an error is returned with the
    /// error kind `WouldBlock`. If an error occurs, then that error is returned.
    ///
    /// Note that unlike `wait`, this function will not attempt to drop stdin.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// #![feature(process_try_wait)]
    ///
    /// use std::io;
    /// use std::process::Command;
    ///
    /// let mut child = Command::new("ls").spawn().unwrap();
    ///
    /// match child.try_wait() {
    ///     Ok(status) => println!("exited with: {}", status),
    ///     Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
    ///         println!("status not ready yet, let's really wait");
    ///         let res = child.wait();
    ///         println!("result: {:?}", res);
    ///     }
    ///     Err(e) => println!("error attempting to wait: {}", e),
    /// }
    /// ```
    #[unstable(feature = "process_try_wait", issue = "38903")]
    pub fn try_wait(&mut self) -> io::Result<ExitStatus> {
        self.handle.try_wait().map(ExitStatus)
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
        Ok(Output {
            status: status,
            stdout: stdout,
            stderr: stderr,
        })
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
/// ```
/// use std::process;
///
/// process::exit(0);
/// ```
///
/// Due to [platform-specific behavior], the exit code for this example will be
/// `0` on Linux, but `256` on Windows:
///
/// ```no_run
/// use std::process;
///
/// process::exit(0x0f00);
/// ```
///
/// [platform-specific behavior]: #platform-specific-behavior
#[stable(feature = "rust1", since = "1.0.0")]
pub fn exit(code: i32) -> ! {
    ::sys_common::cleanup();
    ::sys::os::exit(code)
}

/// Terminates the process in an abnormal fashion.
///
/// The function will never return and will immediately terminate the current
/// process in a platform specific "abnormal" manner.
///
/// Note that because this function never returns, and that it terminates the
/// process, no destructors on the current stack or any other thread's stack
/// will be run. If a clean shutdown is needed it is recommended to only call
/// this function at a known point where there are no more destructors left
/// to run.
#[unstable(feature = "process_abort", issue = "37838")]
pub fn abort() -> ! {
    unsafe { ::sys::abort_internal() };
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use io::prelude::*;

    use io::ErrorKind;
    use str;
    use super::{Command, Output, Stdio};

    // FIXME(#10380) these tests should not all be ignored on android.

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn smoke() {
        let p = if cfg!(target_os = "windows") {
            Command::new("cmd").args(&["/C", "exit 0"]).spawn()
        } else {
            Command::new("true").spawn()
        };
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().unwrap().success());
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn smoke_failure() {
        match Command::new("if-this-is-a-binary-then-the-world-has-ended").spawn() {
            Ok(..) => panic!(),
            Err(..) => {}
        }
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn exit_reported_right() {
        let p = if cfg!(target_os = "windows") {
            Command::new("cmd").args(&["/C", "exit 1"]).spawn()
        } else {
            Command::new("false").spawn()
        };
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().unwrap().code() == Some(1));
        drop(p.wait());
    }

    #[test]
    #[cfg(unix)]
    #[cfg_attr(target_os = "android", ignore)]
    fn signal_reported_right() {
        use os::unix::process::ExitStatusExt;

        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("read a")
                            .stdin(Stdio::piped())
                            .spawn().unwrap();
        p.kill().unwrap();
        match p.wait().unwrap().signal() {
            Some(9) => {},
            result => panic!("not terminated by signal 9 (instead, {:?})",
                             result),
        }
    }

    pub fn run_output(mut cmd: Command) -> String {
        let p = cmd.spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.stdout.is_some());
        let mut ret = String::new();
        p.stdout.as_mut().unwrap().read_to_string(&mut ret).unwrap();
        assert!(p.wait().unwrap().success());
        return ret;
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn stdout_works() {
        if cfg!(target_os = "windows") {
            let mut cmd = Command::new("cmd");
            cmd.args(&["/C", "echo foobar"]).stdout(Stdio::piped());
            assert_eq!(run_output(cmd), "foobar\r\n");
        } else {
            let mut cmd = Command::new("echo");
            cmd.arg("foobar").stdout(Stdio::piped());
            assert_eq!(run_output(cmd), "foobar\n");
        }
    }

    #[test]
    #[cfg_attr(any(windows, target_os = "android"), ignore)]
    fn set_current_dir_works() {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg("pwd")
           .current_dir("/")
           .stdout(Stdio::piped());
        assert_eq!(run_output(cmd), "/\n");
    }

    #[test]
    #[cfg_attr(any(windows, target_os = "android"), ignore)]
    fn stdin_works() {
        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("read line; echo $line")
                            .stdin(Stdio::piped())
                            .stdout(Stdio::piped())
                            .spawn().unwrap();
        p.stdin.as_mut().unwrap().write("foobar".as_bytes()).unwrap();
        drop(p.stdin.take());
        let mut out = String::new();
        p.stdout.as_mut().unwrap().read_to_string(&mut out).unwrap();
        assert!(p.wait().unwrap().success());
        assert_eq!(out, "foobar\n");
    }


    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    #[cfg(unix)]
    fn uid_works() {
        use os::unix::prelude::*;
        use libc;
        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("true")
                            .uid(unsafe { libc::getuid() })
                            .gid(unsafe { libc::getgid() })
                            .spawn().unwrap();
        assert!(p.wait().unwrap().success());
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    #[cfg(unix)]
    fn uid_to_root_fails() {
        use os::unix::prelude::*;
        use libc;

        // if we're already root, this isn't a valid test. Most of the bots run
        // as non-root though (android is an exception).
        if unsafe { libc::getuid() == 0 } { return }
        assert!(Command::new("/bin/ls").uid(0).gid(0).spawn().is_err());
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn test_process_status() {
        let mut status = if cfg!(target_os = "windows") {
            Command::new("cmd").args(&["/C", "exit 1"]).status().unwrap()
        } else {
            Command::new("false").status().unwrap()
        };
        assert!(status.code() == Some(1));

        status = if cfg!(target_os = "windows") {
            Command::new("cmd").args(&["/C", "exit 0"]).status().unwrap()
        } else {
            Command::new("true").status().unwrap()
        };
        assert!(status.success());
    }

    #[test]
    fn test_process_output_fail_to_start() {
        match Command::new("/no-binary-by-this-name-should-exist").output() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::NotFound),
            Ok(..) => panic!()
        }
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn test_process_output_output() {
        let Output {status, stdout, stderr}
             = if cfg!(target_os = "windows") {
                 Command::new("cmd").args(&["/C", "echo hello"]).output().unwrap()
             } else {
                 Command::new("echo").arg("hello").output().unwrap()
             };
        let output_str = str::from_utf8(&stdout).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello");
        assert_eq!(stderr, Vec::new());
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn test_process_output_error() {
        let Output {status, stdout, stderr}
             = if cfg!(target_os = "windows") {
                 Command::new("cmd").args(&["/C", "mkdir ."]).output().unwrap()
             } else {
                 Command::new("mkdir").arg(".").output().unwrap()
             };

        assert!(status.code() == Some(1));
        assert_eq!(stdout, Vec::new());
        assert!(!stderr.is_empty());
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn test_finish_once() {
        let mut prog = if cfg!(target_os = "windows") {
            Command::new("cmd").args(&["/C", "exit 1"]).spawn().unwrap()
        } else {
            Command::new("false").spawn().unwrap()
        };
        assert!(prog.wait().unwrap().code() == Some(1));
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn test_finish_twice() {
        let mut prog = if cfg!(target_os = "windows") {
            Command::new("cmd").args(&["/C", "exit 1"]).spawn().unwrap()
        } else {
            Command::new("false").spawn().unwrap()
        };
        assert!(prog.wait().unwrap().code() == Some(1));
        assert!(prog.wait().unwrap().code() == Some(1));
    }

    #[test]
    #[cfg_attr(target_os = "android", ignore)]
    fn test_wait_with_output_once() {
        let prog = if cfg!(target_os = "windows") {
            Command::new("cmd").args(&["/C", "echo hello"]).stdout(Stdio::piped()).spawn().unwrap()
        } else {
            Command::new("echo").arg("hello").stdout(Stdio::piped()).spawn().unwrap()
        };

        let Output {status, stdout, stderr} = prog.wait_with_output().unwrap();
        let output_str = str::from_utf8(&stdout).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello");
        assert_eq!(stderr, Vec::new());
    }

    #[cfg(all(unix, not(target_os="android")))]
    pub fn env_cmd() -> Command {
        Command::new("env")
    }
    #[cfg(target_os="android")]
    pub fn env_cmd() -> Command {
        let mut cmd = Command::new("/system/bin/sh");
        cmd.arg("-c").arg("set");
        cmd
    }

    #[cfg(windows)]
    pub fn env_cmd() -> Command {
        let mut cmd = Command::new("cmd");
        cmd.arg("/c").arg("set");
        cmd
    }

    #[test]
    fn test_inherit_env() {
        use env;

        let result = env_cmd().output().unwrap();
        let output = String::from_utf8(result.stdout).unwrap();

        for (ref k, ref v) in env::vars() {
            // don't check android RANDOM variables
            if cfg!(target_os = "android") && *k == "RANDOM" {
                continue
            }

            // Windows has hidden environment variables whose names start with
            // equals signs (`=`). Those do not show up in the output of the
            // `set` command.
            assert!((cfg!(windows) && k.starts_with("=")) ||
                    k.starts_with("DYLD") ||
                    output.contains(&format!("{}={}", *k, *v)) ||
                    output.contains(&format!("{}='{}'", *k, *v)),
                    "output doesn't contain `{}={}`\n{}",
                    k, v, output);
        }
    }

    #[test]
    fn test_override_env() {
        use env;

        // In some build environments (such as chrooted Nix builds), `env` can
        // only be found in the explicitly-provided PATH env variable, not in
        // default places such as /bin or /usr/bin. So we need to pass through
        // PATH to our sub-process.
        let mut cmd = env_cmd();
        cmd.env_clear().env("RUN_TEST_NEW_ENV", "123");
        if let Some(p) = env::var_os("PATH") {
            cmd.env("PATH", &p);
        }
        let result = cmd.output().unwrap();
        let output = String::from_utf8_lossy(&result.stdout).to_string();

        assert!(output.contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    }

    #[test]
    fn test_add_to_env() {
        let result = env_cmd().env("RUN_TEST_NEW_ENV", "123").output().unwrap();
        let output = String::from_utf8_lossy(&result.stdout).to_string();

        assert!(output.contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    }

    // Regression tests for #30858.
    #[test]
    fn test_interior_nul_in_progname_is_error() {
        match Command::new("has-some-\0\0s-inside").spawn() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn test_interior_nul_in_arg_is_error() {
        match Command::new("echo").arg("has-some-\0\0s-inside").spawn() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn test_interior_nul_in_args_is_error() {
        match Command::new("echo").args(&["has-some-\0\0s-inside"]).spawn() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn test_interior_nul_in_current_dir_is_error() {
        match Command::new("echo").current_dir("has-some-\0\0s-inside").spawn() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
            Ok(_) => panic!(),
        }
    }

    // Regression tests for #30862.
    #[test]
    fn test_interior_nul_in_env_key_is_error() {
        match env_cmd().env("has-some-\0\0s-inside", "value").spawn() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
            Ok(_) => panic!(),
        }
    }

    #[test]
    fn test_interior_nul_in_env_value_is_error() {
        match env_cmd().env("key", "has-some-\0\0s-inside").spawn() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
            Ok(_) => panic!(),
        }
    }

    /// Test that process creation flags work by debugging a process.
    /// Other creation flags make it hard or impossible to detect
    /// behavioral changes in the process.
    #[test]
    #[cfg(windows)]
    fn test_creation_flags() {
        use os::windows::process::CommandExt;
        use sys::c::{BOOL, DWORD, INFINITE};
        #[repr(C, packed)]
        struct DEBUG_EVENT {
            pub event_code: DWORD,
            pub process_id: DWORD,
            pub thread_id: DWORD,
            // This is a union in the real struct, but we don't
            // need this data for the purposes of this test.
            pub _junk: [u8; 164],
        }

        extern "system" {
            fn WaitForDebugEvent(lpDebugEvent: *mut DEBUG_EVENT, dwMilliseconds: DWORD) -> BOOL;
            fn ContinueDebugEvent(dwProcessId: DWORD, dwThreadId: DWORD,
                                  dwContinueStatus: DWORD) -> BOOL;
        }

        const DEBUG_PROCESS: DWORD = 1;
        const EXIT_PROCESS_DEBUG_EVENT: DWORD = 5;
        const DBG_EXCEPTION_NOT_HANDLED: DWORD = 0x80010001;

        let mut child = Command::new("cmd")
            .creation_flags(DEBUG_PROCESS)
            .stdin(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(b"exit\r\n").unwrap();
        let mut events = 0;
        let mut event = DEBUG_EVENT {
            event_code: 0,
            process_id: 0,
            thread_id: 0,
            _junk: [0; 164],
        };
        loop {
            if unsafe { WaitForDebugEvent(&mut event as *mut DEBUG_EVENT, INFINITE) } == 0 {
                panic!("WaitForDebugEvent failed!");
            }
            events += 1;

            if event.event_code == EXIT_PROCESS_DEBUG_EVENT {
                break;
            }

            if unsafe { ContinueDebugEvent(event.process_id,
                                           event.thread_id,
                                           DBG_EXCEPTION_NOT_HANDLED) } == 0 {
                panic!("ContinueDebugEvent failed!");
            }
        }
        assert!(events > 0);
    }
}
