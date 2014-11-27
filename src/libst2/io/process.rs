// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Bindings for executing child processes

#![allow(experimental)]
#![allow(non_upper_case_globals)]

pub use self::StdioContainer::*;
pub use self::ProcessExit::*;

use prelude::*;

use fmt;
use os;
use io::{IoResult, IoError};
use io;
use libc;
use c_str::CString;
use collections::HashMap;
use hash::Hash;
#[cfg(windows)]
use std::hash::sip::SipState;
use io::pipe::{PipeStream, PipePair};
use path::BytesContainer;

use sys;
use sys::fs::FileDesc;
use sys::process::Process as ProcessImp;

/// Signal a process to exit, without forcibly killing it. Corresponds to
/// SIGTERM on unix platforms.
#[cfg(windows)] pub const PleaseExitSignal: int = 15;
/// Signal a process to exit immediately, forcibly killing it. Corresponds to
/// SIGKILL on unix platforms.
#[cfg(windows)] pub const MustDieSignal: int = 9;
/// Signal a process to exit, without forcibly killing it. Corresponds to
/// SIGTERM on unix platforms.
#[cfg(not(windows))] pub const PleaseExitSignal: int = libc::SIGTERM as int;
/// Signal a process to exit immediately, forcibly killing it. Corresponds to
/// SIGKILL on unix platforms.
#[cfg(not(windows))] pub const MustDieSignal: int = libc::SIGKILL as int;

/// Representation of a running or exited child process.
///
/// This structure is used to represent and manage child processes. A child
/// process is created via the `Command` struct, which configures the spawning
/// process and can itself be constructed using a builder-style interface.
///
/// # Example
///
/// ```should_fail
/// use std::io::Command;
///
/// let mut child = match Command::new("/bin/cat").arg("file.txt").spawn() {
///     Ok(child) => child,
///     Err(e) => panic!("failed to execute child: {}", e),
/// };
///
/// let contents = child.stdout.as_mut().unwrap().read_to_end();
/// assert!(child.wait().unwrap().success());
/// ```
pub struct Process {
    handle: ProcessImp,
    forget: bool,

    /// None until wait() is called.
    exit_code: Option<ProcessExit>,

    /// Manually delivered signal
    exit_signal: Option<int>,

    /// Deadline after which wait() will return
    deadline: u64,

    /// Handle to the child's stdin, if the `stdin` field of this process's
    /// `ProcessConfig` was `CreatePipe`. By default, this handle is `Some`.
    pub stdin: Option<PipeStream>,

    /// Handle to the child's stdout, if the `stdout` field of this process's
    /// `ProcessConfig` was `CreatePipe`. By default, this handle is `Some`.
    pub stdout: Option<PipeStream>,

    /// Handle to the child's stderr, if the `stderr` field of this process's
    /// `ProcessConfig` was `CreatePipe`. By default, this handle is `Some`.
    pub stderr: Option<PipeStream>,
}

/// A representation of environment variable name
/// It compares case-insensitive on Windows and case-sensitive everywhere else.
#[cfg(not(windows))]
#[deriving(PartialEq, Eq, Hash, Clone, Show)]
struct EnvKey(CString);

#[doc(hidden)]
#[cfg(windows)]
#[deriving(Eq, Clone, Show)]
struct EnvKey(CString);

#[cfg(windows)]
impl Hash for EnvKey {
    fn hash(&self, state: &mut SipState) { unimplemented!() }
}

#[cfg(windows)]
impl PartialEq for EnvKey {
    fn eq(&self, other: &EnvKey) -> bool { unimplemented!() }
}

impl BytesContainer for EnvKey {
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] { unimplemented!() }
}

/// A HashMap representation of environment variables.
pub type EnvMap = HashMap<EnvKey, CString>;

/// The `Command` type acts as a process builder, providing fine-grained control
/// over how a new process should be spawned. A default configuration can be
/// generated using `Command::new(program)`, where `program` gives a path to the
/// program to be executed. Additional builder methods allow the configuration
/// to be changed (for example, by adding arguments) prior to spawning:
///
/// ```
/// use std::io::Command;
///
/// let mut process = match Command::new("sh").arg("-c").arg("echo hello").spawn() {
///   Ok(p) => p,
///   Err(e) => panic!("failed to execute process: {}", e),
/// };
///
/// let output = process.stdout.as_mut().unwrap().read_to_end();
/// ```
#[deriving(Clone)]
pub struct Command {
    // The internal data for the builder. Documented by the builder
    // methods below, and serialized into rt::rtio::ProcessConfig.
    program: CString,
    args: Vec<CString>,
    env: Option<EnvMap>,
    cwd: Option<CString>,
    stdin: StdioContainer,
    stdout: StdioContainer,
    stderr: StdioContainer,
    uid: Option<uint>,
    gid: Option<uint>,
    detach: bool,
}

// FIXME (#12938): Until DST lands, we cannot decompose &str into & and str, so
// we cannot usefully take ToCStr arguments by reference (without forcing an
// additional & around &str). So we are instead temporarily adding an instance
// for &Path, so that we can take ToCStr as owned. When DST lands, the &Path
// instance should be removed, and arguments bound by ToCStr should be passed by
// reference. (Here: {new, arg, args, env}.)

impl Command {
    /// Constructs a new `Command` for launching the program at
    /// path `program`, with the following default configuration:
    ///
    /// * No arguments to the program
    /// * Inherit the current process's environment
    /// * Inherit the current process's working directory
    /// * A readable pipe for stdin (file descriptor 0)
    /// * A writeable pipe for stdout and stderr (file descriptors 1 and 2)
    ///
    /// Builder methods are provided to change these defaults and
    /// otherwise configure the process.
    pub fn new<T:ToCStr>(program: T) -> Command { unimplemented!() }

    /// Add an argument to pass to the program.
    pub fn arg<'a, T: ToCStr>(&'a mut self, arg: T) -> &'a mut Command { unimplemented!() }

    /// Add multiple arguments to pass to the program.
    pub fn args<'a, T: ToCStr>(&'a mut self, args: &[T]) -> &'a mut Command { unimplemented!() }
    // Get a mutable borrow of the environment variable map for this `Command`.
    fn get_env_map<'a>(&'a mut self) -> &'a mut  EnvMap { unimplemented!() }

    /// Inserts or updates an environment variable mapping.
    ///
    /// Note that environment variable names are case-insensitive (but case-preserving) on Windows,
    /// and case-sensitive on all other platforms.
    pub fn env<'a, T: ToCStr, U: ToCStr>(&'a mut self, key: T, val: U)
                                         -> &'a mut Command { unimplemented!() }

    /// Removes an environment variable mapping.
    pub fn env_remove<'a, T: ToCStr>(&'a mut self, key: T) -> &'a mut Command { unimplemented!() }

    /// Sets the entire environment map for the child process.
    ///
    /// If the given slice contains multiple instances of an environment
    /// variable, the *rightmost* instance will determine the value.
    pub fn env_set_all<'a, T: ToCStr, U: ToCStr>(&'a mut self, env: &[(T,U)])
                                                 -> &'a mut Command { unimplemented!() }

    /// Set the working directory for the child process.
    pub fn cwd<'a>(&'a mut self, dir: &Path) -> &'a mut Command { unimplemented!() }

    /// Configuration for the child process's stdin handle (file descriptor 0).
    /// Defaults to `CreatePipe(true, false)` so the input can be written to.
    pub fn stdin<'a>(&'a mut self, cfg: StdioContainer) -> &'a mut Command { unimplemented!() }

    /// Configuration for the child process's stdout handle (file descriptor 1).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    pub fn stdout<'a>(&'a mut self, cfg: StdioContainer) -> &'a mut Command { unimplemented!() }

    /// Configuration for the child process's stderr handle (file descriptor 2).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    pub fn stderr<'a>(&'a mut self, cfg: StdioContainer) -> &'a mut Command { unimplemented!() }

    /// Sets the child process's user id. This translates to a `setuid` call in
    /// the child process. Setting this value on windows will cause the spawn to
    /// fail. Failure in the `setuid` call on unix will also cause the spawn to
    /// fail.
    pub fn uid<'a>(&'a mut self, id: uint) -> &'a mut Command { unimplemented!() }

    /// Similar to `uid`, but sets the group id of the child process. This has
    /// the same semantics as the `uid` field.
    pub fn gid<'a>(&'a mut self, id: uint) -> &'a mut Command { unimplemented!() }

    /// Sets the child process to be spawned in a detached state. On unix, this
    /// means that the child is the leader of a new process group.
    pub fn detached<'a>(&'a mut self) -> &'a mut Command { unimplemented!() }

    /// Executes the command as a child process, which is returned.
    pub fn spawn(&self) -> IoResult<Process> { unimplemented!() }

    /// Executes the command as a child process, waiting for it to finish and
    /// collecting all of its output.
    ///
    /// # Example
    ///
    /// ```
    /// use std::io::Command;
    ///
    /// let output = match Command::new("cat").arg("foot.txt").output() {
    ///     Ok(output) => output,
    ///     Err(e) => panic!("failed to execute process: {}", e),
    /// };
    ///
    /// println!("status: {}", output.status);
    /// println!("stdout: {}", String::from_utf8_lossy(output.output.as_slice()));
    /// println!("stderr: {}", String::from_utf8_lossy(output.error.as_slice()));
    /// ```
    pub fn output(&self) -> IoResult<ProcessOutput> { unimplemented!() }

    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its exit status.
    ///
    /// # Example
    ///
    /// ```
    /// use std::io::Command;
    ///
    /// let status = match Command::new("ls").status() {
    ///     Ok(status) => status,
    ///     Err(e) => panic!("failed to execute process: {}", e),
    /// };
    ///
    /// println!("process exited with: {}", status);
    /// ```
    pub fn status(&self) -> IoResult<ProcessExit> { unimplemented!() }
}

impl fmt::Show for Command {
    /// Format the program and arguments of a Command for display. Any
    /// non-utf8 data is lossily converted using the utf8 replacement
    /// character.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

fn setup_io(io: StdioContainer) -> IoResult<(Option<PipeStream>, Option<PipeStream>)> { unimplemented!() }

// Allow the sys module to get access to the Command state
impl sys::process::ProcessConfig<EnvKey, CString> for Command {
    fn program(&self) -> &CString { unimplemented!() }
    fn args(&self) -> &[CString] { unimplemented!() }
    fn env(&self) -> Option<&EnvMap> { unimplemented!() }
    fn cwd(&self) -> Option<&CString> { unimplemented!() }
    fn uid(&self) -> Option<uint> { unimplemented!() }
    fn gid(&self) -> Option<uint> { unimplemented!() }
    fn detach(&self) -> bool { unimplemented!() }

}

/// The output of a finished process.
#[deriving(PartialEq, Eq, Clone)]
pub struct ProcessOutput {
    /// The status (exit code) of the process.
    pub status: ProcessExit,
    /// The data that the process wrote to stdout.
    pub output: Vec<u8>,
    /// The data that the process wrote to stderr.
    pub error: Vec<u8>,
}

/// Describes what to do with a standard io stream for a child process.
#[deriving(Clone)]
pub enum StdioContainer {
    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    Ignored,

    /// The specified file descriptor is inherited for the stream which it is
    /// specified for. Ownership of the file descriptor is *not* taken, so the
    /// caller must clean it up.
    InheritFd(libc::c_int),

    /// Creates a pipe for the specified file descriptor which will be created
    /// when the process is spawned.
    ///
    /// The first boolean argument is whether the pipe is readable, and the
    /// second is whether it is writable. These properties are from the view of
    /// the *child* process, not the parent process.
    CreatePipe(bool /* readable */, bool /* writable */),
}

/// Describes the result of a process after it has terminated.
/// Note that Windows have no signals, so the result is usually ExitStatus.
#[deriving(PartialEq, Eq, Clone)]
pub enum ProcessExit {
    /// Normal termination with an exit status.
    ExitStatus(int),

    /// Termination by signal, with the signal number.
    ExitSignal(int),
}

impl fmt::Show for ProcessExit {
    /// Format a ProcessExit enum, to nicely present the information.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

impl ProcessExit {
    /// Was termination successful? Signal termination not considered a success,
    /// and success is defined as a zero exit status.
    pub fn success(&self) -> bool { unimplemented!() }

    /// Checks whether this ProcessExit matches the given exit status.
    /// Termination by signal will never match an exit code.
    pub fn matches_exit_status(&self, wanted: int) -> bool { unimplemented!() }
}

impl Process {
    /// Sends `signal` to another process in the system identified by `id`.
    ///
    /// Note that windows doesn't quite have the same model as unix, so some
    /// unix signals are mapped to windows signals. Notably, unix termination
    /// signals (SIGTERM/SIGKILL/SIGINT) are translated to `TerminateProcess`.
    ///
    /// Additionally, a signal number of 0 can check for existence of the target
    /// process. Note, though, that on some platforms signals will continue to
    /// be successfully delivered if the child has exited, but not yet been
    /// reaped.
    pub fn kill(id: libc::pid_t, signal: int) -> IoResult<()> { unimplemented!() }

    /// Returns the process id of this child process
    pub fn id(&self) -> libc::pid_t { unimplemented!() }

    /// Sends the specified signal to the child process, returning whether the
    /// signal could be delivered or not.
    ///
    /// Note that signal 0 is interpreted as a poll to check whether the child
    /// process is still alive or not. If an error is returned, then the child
    /// process has exited.
    ///
    /// On some unix platforms signals will continue to be received after a
    /// child has exited but not yet been reaped. In order to report the status
    /// of signal delivery correctly, unix implementations may invoke
    /// `waitpid()` with `WNOHANG` in order to reap the child as necessary.
    ///
    /// # Errors
    ///
    /// If the signal delivery fails, the corresponding error is returned.
    pub fn signal(&mut self, signal: int) -> IoResult<()> { unimplemented!() }

    /// Sends a signal to this child requesting that it exits. This is
    /// equivalent to sending a SIGTERM on unix platforms.
    pub fn signal_exit(&mut self) -> IoResult<()> { unimplemented!() }

    /// Sends a signal to this child forcing it to exit. This is equivalent to
    /// sending a SIGKILL on unix platforms.
    pub fn signal_kill(&mut self) -> IoResult<()> { unimplemented!() }

    /// Wait for the child to exit completely, returning the status that it
    /// exited with. This function will continue to have the same return value
    /// after it has been called at least once.
    ///
    /// The stdin handle to the child process will be closed before waiting.
    ///
    /// # Errors
    ///
    /// This function can fail if a timeout was previously specified via
    /// `set_timeout` and the timeout expires before the child exits.
    pub fn wait(&mut self) -> IoResult<ProcessExit> { unimplemented!() }

    /// Sets a timeout, in milliseconds, for future calls to wait().
    ///
    /// The argument specified is a relative distance into the future, in
    /// milliseconds, after which any call to wait() will return immediately
    /// with a timeout error, and all future calls to wait() will not block.
    ///
    /// A value of `None` will clear any previous timeout, and a value of `Some`
    /// will override any previously set timeout.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #![allow(experimental)]
    /// use std::io::{Command, IoResult};
    /// use std::io::process::ProcessExit;
    ///
    /// fn run_gracefully(prog: &str) -> IoResult<ProcessExit> {
    ///     let mut p = try!(Command::new("long-running-process").spawn());
    ///
    ///     // give the process 10 seconds to finish completely
    ///     p.set_timeout(Some(10_000));
    ///     match p.wait() {
    ///         Ok(status) => return Ok(status),
    ///         Err(..) => {}
    ///     }
    ///
    ///     // Attempt to exit gracefully, but don't wait for it too long
    ///     try!(p.signal_exit());
    ///     p.set_timeout(Some(1_000));
    ///     match p.wait() {
    ///         Ok(status) => return Ok(status),
    ///         Err(..) => {}
    ///     }
    ///
    ///     // Well, we did our best, forcefully kill the process
    ///     try!(p.signal_kill());
    ///     p.set_timeout(None);
    ///     p.wait()
    /// }
    /// ```
    #[experimental = "the type of the timeout is likely to change"]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) { unimplemented!() }

    /// Simultaneously wait for the child to exit and collect all remaining
    /// output on the stdout/stderr handles, returning a `ProcessOutput`
    /// instance.
    ///
    /// The stdin handle to the child is closed before waiting.
    ///
    /// # Errors
    ///
    /// This function can fail for any of the same reasons that `wait()` can
    /// fail.
    pub fn wait_with_output(mut self) -> IoResult<ProcessOutput> { unimplemented!() }

    /// Forgets this process, allowing it to outlive the parent
    ///
    /// This function will forcefully prevent calling `wait()` on the child
    /// process in the destructor, allowing the child to outlive the
    /// parent. Note that this operation can easily lead to leaking the
    /// resources of the child process, so care must be taken when
    /// invoking this method.
    pub fn forget(mut self) { unimplemented!() }
}

impl Drop for Process {
    fn drop(&mut self) { unimplemented!() }
}
