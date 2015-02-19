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

#![allow(non_upper_case_globals)]

pub use self::StdioContainer::*;
pub use self::ProcessExit::*;

use prelude::v1::*;

use collections::HashMap;
use ffi::CString;
use fmt;
use old_io::pipe::{PipeStream, PipePair};
use old_io::{IoResult, IoError};
use old_io;
use libc;
use os;
use old_path::BytesContainer;
use sync::mpsc::{channel, Receiver};
use sys::fs::FileDesc;
use sys::process::Process as ProcessImp;
use sys;
use thread;

#[cfg(windows)] use hash;
#[cfg(windows)] use str;

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
/// use std::old_io::Command;
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
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct EnvKey(CString);

#[doc(hidden)]
#[cfg(windows)]
#[derive(Eq, Clone, Debug)]
struct EnvKey(CString);

#[cfg(all(windows, stage0))]
impl<H: hash::Writer + hash::Hasher> hash::Hash<H> for EnvKey {
    fn hash(&self, state: &mut H) {
        let &EnvKey(ref x) = self;
        match str::from_utf8(x.as_bytes()) {
            Ok(s) => for ch in s.chars() {
                (ch as u8 as char).to_lowercase().hash(state);
            },
            Err(..) => x.hash(state)
        }
    }
}
#[cfg(all(windows, not(stage0)))]
impl hash::Hash for EnvKey {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        let &EnvKey(ref x) = self;
        match str::from_utf8(x.as_bytes()) {
            Ok(s) => for ch in s.chars() {
                (ch as u8 as char).to_lowercase().hash(state);
            },
            Err(..) => x.hash(state)
        }
    }
}

#[cfg(windows)]
impl PartialEq for EnvKey {
    fn eq(&self, other: &EnvKey) -> bool {
        let &EnvKey(ref x) = self;
        let &EnvKey(ref y) = other;
        match (str::from_utf8(x.as_bytes()), str::from_utf8(y.as_bytes())) {
            (Ok(xs), Ok(ys)) => {
                if xs.len() != ys.len() {
                    return false
                } else {
                    for (xch, ych) in xs.chars().zip(ys.chars()) {
                        if xch.to_lowercase() != ych.to_lowercase() {
                            return false;
                        }
                    }
                    return true;
                }
            },
            // If either is not a valid utf8 string, just compare them byte-wise
            _ => return x.eq(y)
        }
    }
}

impl BytesContainer for EnvKey {
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] {
        let &EnvKey(ref k) = self;
        k.container_as_bytes()
    }
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
/// use std::old_io::Command;
///
/// let mut process = match Command::new("sh").arg("-c").arg("echo hello").spawn() {
///   Ok(p) => p,
///   Err(e) => panic!("failed to execute process: {}", e),
/// };
///
/// let output = process.stdout.as_mut().unwrap().read_to_end();
/// ```
#[derive(Clone)]
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
// we cannot usefully take BytesContainer arguments by reference (without forcing an
// additional & around &str). So we are instead temporarily adding an instance
// for &Path, so that we can take BytesContainer as owned. When DST lands, the &Path
// instance should be removed, and arguments bound by BytesContainer should be passed by
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
    pub fn new<T: BytesContainer>(program: T) -> Command {
        Command {
            program: CString::new(program.container_as_bytes()).unwrap(),
            args: Vec::new(),
            env: None,
            cwd: None,
            stdin: CreatePipe(true, false),
            stdout: CreatePipe(false, true),
            stderr: CreatePipe(false, true),
            uid: None,
            gid: None,
            detach: false,
        }
    }

    /// Add an argument to pass to the program.
    pub fn arg<'a, T: BytesContainer>(&'a mut self, arg: T) -> &'a mut Command {
        self.args.push(CString::new(arg.container_as_bytes()).unwrap());
        self
    }

    /// Add multiple arguments to pass to the program.
    pub fn args<'a, T: BytesContainer>(&'a mut self, args: &[T]) -> &'a mut Command {
        self.args.extend(args.iter().map(|arg| {
            CString::new(arg.container_as_bytes()).unwrap()
        }));
        self
    }
    // Get a mutable borrow of the environment variable map for this `Command`.
    #[allow(deprecated)]
    fn get_env_map<'a>(&'a mut self) -> &'a mut EnvMap {
        match self.env {
            Some(ref mut map) => map,
            None => {
                // if the env is currently just inheriting from the parent's,
                // materialize the parent's env into a hashtable.
                self.env = Some(os::env_as_bytes().into_iter().map(|(k, v)| {
                    (EnvKey(CString::new(k).unwrap()),
                     CString::new(v).unwrap())
                }).collect());
                self.env.as_mut().unwrap()
            }
        }
    }

    /// Inserts or updates an environment variable mapping.
    ///
    /// Note that environment variable names are case-insensitive (but case-preserving) on Windows,
    /// and case-sensitive on all other platforms.
    pub fn env<'a, T, U>(&'a mut self, key: T, val: U)
                         -> &'a mut Command
                         where T: BytesContainer, U: BytesContainer {
        let key = EnvKey(CString::new(key.container_as_bytes()).unwrap());
        let val = CString::new(val.container_as_bytes()).unwrap();
        self.get_env_map().insert(key, val);
        self
    }

    /// Removes an environment variable mapping.
    pub fn env_remove<'a, T>(&'a mut self, key: T) -> &'a mut Command
                             where T: BytesContainer {
        let key = EnvKey(CString::new(key.container_as_bytes()).unwrap());
        self.get_env_map().remove(&key);
        self
    }

    /// Sets the entire environment map for the child process.
    ///
    /// If the given slice contains multiple instances of an environment
    /// variable, the *rightmost* instance will determine the value.
    pub fn env_set_all<'a, T, U>(&'a mut self, env: &[(T,U)])
                                 -> &'a mut Command
                                 where T: BytesContainer, U: BytesContainer {
        self.env = Some(env.iter().map(|&(ref k, ref v)| {
            (EnvKey(CString::new(k.container_as_bytes()).unwrap()),
             CString::new(v.container_as_bytes()).unwrap())
        }).collect());
        self
    }

    /// Set the working directory for the child process.
    pub fn cwd<'a>(&'a mut self, dir: &Path) -> &'a mut Command {
        self.cwd = Some(CString::new(dir.as_vec()).unwrap());
        self
    }

    /// Configuration for the child process's stdin handle (file descriptor 0).
    /// Defaults to `CreatePipe(true, false)` so the input can be written to.
    pub fn stdin<'a>(&'a mut self, cfg: StdioContainer) -> &'a mut Command {
        self.stdin = cfg;
        self
    }

    /// Configuration for the child process's stdout handle (file descriptor 1).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    pub fn stdout<'a>(&'a mut self, cfg: StdioContainer) -> &'a mut Command {
        self.stdout = cfg;
        self
    }

    /// Configuration for the child process's stderr handle (file descriptor 2).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    pub fn stderr<'a>(&'a mut self, cfg: StdioContainer) -> &'a mut Command {
        self.stderr = cfg;
        self
    }

    /// Sets the child process's user id. This translates to a `setuid` call in
    /// the child process. Setting this value on windows will cause the spawn to
    /// fail. Failure in the `setuid` call on unix will also cause the spawn to
    /// fail.
    pub fn uid<'a>(&'a mut self, id: uint) -> &'a mut Command {
        self.uid = Some(id);
        self
    }

    /// Similar to `uid`, but sets the group id of the child process. This has
    /// the same semantics as the `uid` field.
    pub fn gid<'a>(&'a mut self, id: uint) -> &'a mut Command {
        self.gid = Some(id);
        self
    }

    /// Sets the child process to be spawned in a detached state. On unix, this
    /// means that the child is the leader of a new process group.
    pub fn detached<'a>(&'a mut self) -> &'a mut Command {
        self.detach = true;
        self
    }

    /// Executes the command as a child process, which is returned.
    pub fn spawn(&self) -> IoResult<Process> {
        let (their_stdin, our_stdin) = try!(setup_io(self.stdin));
        let (their_stdout, our_stdout) = try!(setup_io(self.stdout));
        let (their_stderr, our_stderr) = try!(setup_io(self.stderr));

        match ProcessImp::spawn(self, their_stdin, their_stdout, their_stderr) {
            Err(e) => Err(e),
            Ok(handle) => Ok(Process {
                handle: handle,
                forget: false,
                exit_code: None,
                exit_signal: None,
                deadline: 0,
                stdin: our_stdin,
                stdout: our_stdout,
                stderr: our_stderr,
            })
        }
    }

    /// Executes the command as a child process, waiting for it to finish and
    /// collecting all of its output.
    ///
    /// # Example
    ///
    /// ```
    /// use std::old_io::Command;
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
    pub fn output(&self) -> IoResult<ProcessOutput> {
        self.spawn().and_then(|p| p.wait_with_output())
    }

    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its exit status.
    ///
    /// # Example
    ///
    /// ```
    /// use std::old_io::Command;
    ///
    /// let status = match Command::new("ls").status() {
    ///     Ok(status) => status,
    ///     Err(e) => panic!("failed to execute process: {}", e),
    /// };
    ///
    /// println!("process exited with: {}", status);
    /// ```
    pub fn status(&self) -> IoResult<ProcessExit> {
        self.spawn().and_then(|mut p| p.wait())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Command {
    /// Format the program and arguments of a Command for display. Any
    /// non-utf8 data is lossily converted using the utf8 replacement
    /// character.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{:?}", self.program));
        for arg in &self.args {
            try!(write!(f, " '{:?}'", arg));
        }
        Ok(())
    }
}

fn setup_io(io: StdioContainer) -> IoResult<(Option<PipeStream>, Option<PipeStream>)> {
    let ours;
    let theirs;
    match io {
        Ignored => {
            theirs = None;
            ours = None;
        }
        InheritFd(fd) => {
            theirs = Some(PipeStream::from_filedesc(FileDesc::new(fd, false)));
            ours = None;
        }
        CreatePipe(readable, _writable) => {
            let PipePair { reader, writer } = try!(PipeStream::pair());
            if readable {
                theirs = Some(reader);
                ours = Some(writer);
            } else {
                theirs = Some(writer);
                ours = Some(reader);
            }
        }
    }
    Ok((theirs, ours))
}

// Allow the sys module to get access to the Command state
impl sys::process::ProcessConfig<EnvKey, CString> for Command {
    fn program(&self) -> &CString {
        &self.program
    }
    fn args(&self) -> &[CString] {
        &self.args
    }
    fn env(&self) -> Option<&EnvMap> {
        self.env.as_ref()
    }
    fn cwd(&self) -> Option<&CString> {
        self.cwd.as_ref()
    }
    fn uid(&self) -> Option<uint> {
        self.uid.clone()
    }
    fn gid(&self) -> Option<uint> {
        self.gid.clone()
    }
    fn detach(&self) -> bool {
        self.detach
    }

}

/// The output of a finished process.
#[derive(PartialEq, Eq, Clone)]
pub struct ProcessOutput {
    /// The status (exit code) of the process.
    pub status: ProcessExit,
    /// The data that the process wrote to stdout.
    pub output: Vec<u8>,
    /// The data that the process wrote to stderr.
    pub error: Vec<u8>,
}

/// Describes what to do with a standard io stream for a child process.
#[derive(Clone, Copy)]
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
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ProcessExit {
    /// Normal termination with an exit status.
    ExitStatus(int),

    /// Termination by signal, with the signal number.
    ExitSignal(int),
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ProcessExit {
    /// Format a ProcessExit enum, to nicely present the information.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ExitStatus(code) =>  write!(f, "exit code: {}", code),
            ExitSignal(code) =>  write!(f, "signal: {}", code),
        }
    }
}

impl ProcessExit {
    /// Was termination successful? Signal termination not considered a success,
    /// and success is defined as a zero exit status.
    pub fn success(&self) -> bool {
        return self.matches_exit_status(0);
    }

    /// Checks whether this ProcessExit matches the given exit status.
    /// Termination by signal will never match an exit code.
    pub fn matches_exit_status(&self, wanted: int) -> bool {
        *self == ExitStatus(wanted)
    }
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
    pub fn kill(id: libc::pid_t, signal: int) -> IoResult<()> {
        unsafe { ProcessImp::killpid(id, signal) }
    }

    /// Returns the process id of this child process
    pub fn id(&self) -> libc::pid_t { self.handle.id() }

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
    pub fn signal(&mut self, signal: int) -> IoResult<()> {
        #[cfg(unix)] fn collect_status(p: &mut Process) {
            // On Linux (and possibly other unices), a process that has exited will
            // continue to accept signals because it is "defunct". The delivery of
            // signals will only fail once the child has been reaped. For this
            // reason, if the process hasn't exited yet, then we attempt to collect
            // their status with WNOHANG.
            if p.exit_code.is_none() {
                match p.handle.try_wait() {
                    Some(code) => { p.exit_code = Some(code); }
                    None => {}
                }
            }
        }
        #[cfg(windows)] fn collect_status(_p: &mut Process) {}

        collect_status(self);

        // if the process has finished, and therefore had waitpid called,
        // and we kill it, then on unix we might ending up killing a
        // newer process that happens to have the same (re-used) id
        if self.exit_code.is_some() {
            return Err(IoError {
                kind: old_io::InvalidInput,
                desc: "invalid argument: can't kill an exited process",
                detail: None,
            })
        }

        // A successfully delivered signal that isn't 0 (just a poll for being
        // alive) is recorded for windows (see wait())
        match unsafe { self.handle.kill(signal) } {
            Ok(()) if signal == 0 => Ok(()),
            Ok(()) => { self.exit_signal = Some(signal); Ok(()) }
            Err(e) => Err(e),
        }

    }

    /// Sends a signal to this child requesting that it exits. This is
    /// equivalent to sending a SIGTERM on unix platforms.
    pub fn signal_exit(&mut self) -> IoResult<()> {
        self.signal(PleaseExitSignal)
    }

    /// Sends a signal to this child forcing it to exit. This is equivalent to
    /// sending a SIGKILL on unix platforms.
    pub fn signal_kill(&mut self) -> IoResult<()> {
        self.signal(MustDieSignal)
    }

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
    pub fn wait(&mut self) -> IoResult<ProcessExit> {
        drop(self.stdin.take());
        match self.exit_code {
            Some(code) => Ok(code),
            None => {
                let code = try!(self.handle.wait(self.deadline));
                // On windows, waitpid will never return a signal. If a signal
                // was successfully delivered to the process, however, we can
                // consider it as having died via a signal.
                let code = match self.exit_signal {
                    None => code,
                    Some(signal) if cfg!(windows) => ExitSignal(signal),
                    Some(..) => code,
                };
                self.exit_code = Some(code);
                Ok(code)
            }
        }
    }

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
    /// use std::old_io::{Command, IoResult};
    /// use std::old_io::process::ProcessExit;
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
    #[unstable(feature = "io",
               reason = "the type of the timeout is likely to change")]
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) {
        self.deadline = timeout_ms.map(|i| i + sys::timer::now()).unwrap_or(0);
    }

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
    pub fn wait_with_output(mut self) -> IoResult<ProcessOutput> {
        drop(self.stdin.take());
        fn read(stream: Option<old_io::PipeStream>) -> Receiver<IoResult<Vec<u8>>> {
            let (tx, rx) = channel();
            match stream {
                Some(stream) => {
                    thread::spawn(move || {
                        let mut stream = stream;
                        tx.send(stream.read_to_end()).unwrap();
                    });
                }
                None => tx.send(Ok(Vec::new())).unwrap()
            }
            rx
        }
        let stdout = read(self.stdout.take());
        let stderr = read(self.stderr.take());

        let status = try!(self.wait());

        Ok(ProcessOutput {
            status: status,
            output: stdout.recv().unwrap().unwrap_or(Vec::new()),
            error:  stderr.recv().unwrap().unwrap_or(Vec::new()),
        })
    }

    /// Forgets this process, allowing it to outlive the parent
    ///
    /// This function will forcefully prevent calling `wait()` on the child
    /// process in the destructor, allowing the child to outlive the
    /// parent. Note that this operation can easily lead to leaking the
    /// resources of the child process, so care must be taken when
    /// invoking this method.
    pub fn forget(mut self) {
        self.forget = true;
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if self.forget { return }

        // Close all I/O before exiting to ensure that the child doesn't wait
        // forever to print some text or something similar.
        drop(self.stdin.take());
        drop(self.stdout.take());
        drop(self.stderr.take());

        self.set_timeout(None);
        let _ = self.wait().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use old_io::{Truncate, Write, TimedOut, timer, process, FileNotFound};
    use prelude::v1::{Ok, Err, range, drop, Some, None, Vec};
    use prelude::v1::{Path, String, Reader, Writer, Clone};
    use prelude::v1::{SliceExt, Str, StrExt, AsSlice, ToString, GenericPath};
    use old_io::fs::PathExtensions;
    use old_io::timer::*;
    use rt::running_on_valgrind;
    use str;
    use super::{CreatePipe};
    use super::{InheritFd, Process, PleaseExitSignal, Command, ProcessOutput};
    use sync::mpsc::channel;
    use thread;
    use time::Duration;

    // FIXME(#10380) these tests should not all be ignored on android.

    #[cfg(not(target_os="android"))]
    #[test]
    fn smoke() {
        let p = Command::new("true").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().unwrap().success());
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn smoke_failure() {
        match Command::new("if-this-is-a-binary-then-the-world-has-ended").spawn() {
            Ok(..) => panic!(),
            Err(..) => {}
        }
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn exit_reported_right() {
        let p = Command::new("false").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().unwrap().matches_exit_status(1));
        drop(p.wait().clone());
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn signal_reported_right() {
        let p = Command::new("/bin/sh").arg("-c").arg("kill -9 $$").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        match p.wait().unwrap() {
            process::ExitSignal(9) => {},
            result => panic!("not terminated by signal 9 (instead, {})", result),
        }
    }

    pub fn read_all(input: &mut Reader) -> String {
        input.read_to_string().unwrap()
    }

    pub fn run_output(cmd: Command) -> String {
        let p = cmd.spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.stdout.is_some());
        let ret = read_all(p.stdout.as_mut().unwrap() as &mut Reader);
        assert!(p.wait().unwrap().success());
        return ret;
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn stdout_works() {
        let mut cmd = Command::new("echo");
        cmd.arg("foobar").stdout(CreatePipe(false, true));
        assert_eq!(run_output(cmd), "foobar\n");
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn set_cwd_works() {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg("pwd")
           .cwd(&Path::new("/"))
           .stdout(CreatePipe(false, true));
        assert_eq!(run_output(cmd), "/\n");
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn stdin_works() {
        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("read line; echo $line")
                            .stdin(CreatePipe(true, false))
                            .stdout(CreatePipe(false, true))
                            .spawn().unwrap();
        p.stdin.as_mut().unwrap().write("foobar".as_bytes()).unwrap();
        drop(p.stdin.take());
        let out = read_all(p.stdout.as_mut().unwrap() as &mut Reader);
        assert!(p.wait().unwrap().success());
        assert_eq!(out, "foobar\n");
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn detach_works() {
        let mut p = Command::new("true").detached().spawn().unwrap();
        assert!(p.wait().unwrap().success());
    }

    #[cfg(windows)]
    #[test]
    fn uid_fails_on_windows() {
        assert!(Command::new("test").uid(10).spawn().is_err());
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn uid_works() {
        use libc;
        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("true")
                            .uid(unsafe { libc::getuid() as uint })
                            .gid(unsafe { libc::getgid() as uint })
                            .spawn().unwrap();
        assert!(p.wait().unwrap().success());
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn uid_to_root_fails() {
        use libc;

        // if we're already root, this isn't a valid test. Most of the bots run
        // as non-root though (android is an exception).
        if unsafe { libc::getuid() == 0 } { return }
        assert!(Command::new("/bin/ls").uid(0).gid(0).spawn().is_err());
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_process_status() {
        let mut status = Command::new("false").status().unwrap();
        assert!(status.matches_exit_status(1));

        status = Command::new("true").status().unwrap();
        assert!(status.success());
    }

    #[test]
    fn test_process_output_fail_to_start() {
        match Command::new("/no-binary-by-this-name-should-exist").output() {
            Err(e) => assert_eq!(e.kind, FileNotFound),
            Ok(..) => panic!()
        }
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_process_output_output() {
        let ProcessOutput {status, output, error}
             = Command::new("echo").arg("hello").output().unwrap();
        let output_str = str::from_utf8(&output).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_process_output_error() {
        let ProcessOutput {status, output, error}
             = Command::new("mkdir").arg(".").output().unwrap();

        assert!(status.matches_exit_status(1));
        assert_eq!(output, Vec::new());
        assert!(!error.is_empty());
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_finish_once() {
        let mut prog = Command::new("false").spawn().unwrap();
        assert!(prog.wait().unwrap().matches_exit_status(1));
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_finish_twice() {
        let mut prog = Command::new("false").spawn().unwrap();
        assert!(prog.wait().unwrap().matches_exit_status(1));
        assert!(prog.wait().unwrap().matches_exit_status(1));
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_wait_with_output_once() {
        let prog = Command::new("echo").arg("hello").spawn().unwrap();
        let ProcessOutput {status, output, error} = prog.wait_with_output().unwrap();
        let output_str = str::from_utf8(&output).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }
    }

    #[cfg(all(unix, not(target_os="android")))]
    pub fn pwd_cmd() -> Command {
        Command::new("pwd")
    }
    #[cfg(target_os="android")]
    pub fn pwd_cmd() -> Command {
        let mut cmd = Command::new("/system/bin/sh");
        cmd.arg("-c").arg("pwd");
        cmd
    }

    #[cfg(windows)]
    pub fn pwd_cmd() -> Command {
        let mut cmd = Command::new("cmd");
        cmd.arg("/c").arg("cd");
        cmd
    }

    #[test]
    fn test_keep_current_working_dir() {
        use os;
        let prog = pwd_cmd().spawn().unwrap();

        let output = String::from_utf8(prog.wait_with_output().unwrap().output).unwrap();
        let parent_dir = os::getcwd().unwrap();
        let child_dir = Path::new(output.trim());

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.unstable.device, child_stat.unstable.device);
        assert_eq!(parent_stat.unstable.inode, child_stat.unstable.inode);
    }

    #[test]
    fn test_change_working_directory() {
        use os;
        // test changing to the parent of os::getcwd() because we know
        // the path exists (and os::getcwd() is not expected to be root)
        let parent_dir = os::getcwd().unwrap().dir_path();
        let prog = pwd_cmd().cwd(&parent_dir).spawn().unwrap();

        let output = String::from_utf8(prog.wait_with_output().unwrap().output).unwrap();
        let child_dir = Path::new(output.trim());

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.unstable.device, child_stat.unstable.device);
        assert_eq!(parent_stat.unstable.inode, child_stat.unstable.inode);
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

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_inherit_env() {
        use os;
        if running_on_valgrind() { return; }

        let prog = env_cmd().spawn().unwrap();
        let output = String::from_utf8(prog.wait_with_output().unwrap().output).unwrap();

        let r = os::env();
        for &(ref k, ref v) in &r {
            // don't check windows magical empty-named variables
            assert!(k.is_empty() ||
                    output.contains(&format!("{}={}", *k, *v)),
                    "output doesn't contain `{}={}`\n{}",
                    k, v, output);
        }
    }
    #[cfg(target_os="android")]
    #[test]
    fn test_inherit_env() {
        use os;
        if running_on_valgrind() { return; }

        let mut prog = env_cmd().spawn().unwrap();
        let output = String::from_utf8(prog.wait_with_output().unwrap().output).unwrap();

        let r = os::env();
        for &(ref k, ref v) in &r {
            // don't check android RANDOM variables
            if *k != "RANDOM".to_string() {
                assert!(output.contains(&format!("{}={}",
                                                 *k,
                                                 *v)) ||
                        output.contains(&format!("{}=\'{}\'",
                                                 *k,
                                                 *v)));
            }
        }
    }

    #[test]
    fn test_override_env() {
        use os;

        // In some build environments (such as chrooted Nix builds), `env` can
        // only be found in the explicitly-provided PATH env variable, not in
        // default places such as /bin or /usr/bin. So we need to pass through
        // PATH to our sub-process.
        let path_val: String;
        let mut new_env = vec![("RUN_TEST_NEW_ENV", "123")];
        match os::getenv("PATH") {
            None => {}
            Some(val) => {
                path_val = val;
                new_env.push(("PATH", &path_val))
            }
        }

        let prog = env_cmd().env_set_all(&new_env).spawn().unwrap();
        let result = prog.wait_with_output().unwrap();
        let output = String::from_utf8_lossy(&result.output).to_string();

        assert!(output.contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    }

    #[test]
    fn test_add_to_env() {
        let prog = env_cmd().env("RUN_TEST_NEW_ENV", "123").spawn().unwrap();
        let result = prog.wait_with_output().unwrap();
        let output = String::from_utf8_lossy(&result.output).to_string();

        assert!(output.contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    }

    #[cfg(unix)]
    pub fn sleeper() -> Process {
        Command::new("sleep").arg("1000").spawn().unwrap()
    }
    #[cfg(windows)]
    pub fn sleeper() -> Process {
        // There's a `timeout` command on windows, but it doesn't like having
        // its output piped, so instead just ping ourselves a few times with
        // gaps in between so we're sure this process is alive for awhile
        Command::new("ping").arg("127.0.0.1").arg("-n").arg("1000").spawn().unwrap()
    }

    #[test]
    fn test_kill() {
        let mut p = sleeper();
        Process::kill(p.id(), PleaseExitSignal).unwrap();
        assert!(!p.wait().unwrap().success());
    }

    #[test]
    fn test_exists() {
        let mut p = sleeper();
        assert!(Process::kill(p.id(), 0).is_ok());
        p.signal_kill().unwrap();
        assert!(!p.wait().unwrap().success());
    }

    #[test]
    fn test_zero() {
        let mut p = sleeper();
        p.signal_kill().unwrap();
        for _ in 0..20 {
            if p.signal(0).is_err() {
                assert!(!p.wait().unwrap().success());
                return
            }
            timer::sleep(Duration::milliseconds(100));
        }
        panic!("never saw the child go away");
    }

    #[test]
    fn wait_timeout() {
        let mut p = sleeper();
        p.set_timeout(Some(10));
        assert_eq!(p.wait().err().unwrap().kind, TimedOut);
        assert_eq!(p.wait().err().unwrap().kind, TimedOut);
        p.signal_kill().unwrap();
        p.set_timeout(None);
        assert!(p.wait().is_ok());
    }

    #[test]
    fn wait_timeout2() {
        let (tx, rx) = channel();
        let tx2 = tx.clone();
        let _t = thread::spawn(move|| {
            let mut p = sleeper();
            p.set_timeout(Some(10));
            assert_eq!(p.wait().err().unwrap().kind, TimedOut);
            p.signal_kill().unwrap();
            tx.send(()).unwrap();
        });
        let _t = thread::spawn(move|| {
            let mut p = sleeper();
            p.set_timeout(Some(10));
            assert_eq!(p.wait().err().unwrap().kind, TimedOut);
            p.signal_kill().unwrap();
            tx2.send(()).unwrap();
        });
        rx.recv().unwrap();
        rx.recv().unwrap();
    }

    #[test]
    fn forget() {
        let p = sleeper();
        let id = p.id();
        p.forget();
        assert!(Process::kill(id, 0).is_ok());
        assert!(Process::kill(id, PleaseExitSignal).is_ok());
    }

    #[test]
    fn dont_close_fd_on_command_spawn() {
        use sys::fs;

        let path = if cfg!(windows) {
            Path::new("NUL")
        } else {
            Path::new("/dev/null")
        };

        let fdes = match fs::open(&path, Truncate, Write) {
            Ok(f) => f,
            Err(_) => panic!("failed to open file descriptor"),
        };

        let mut cmd = pwd_cmd();
        let _ = cmd.stdout(InheritFd(fdes.fd()));
        assert!(cmd.status().unwrap().success());
        assert!(fdes.write("extra write\n".as_bytes()).is_ok());
    }

    #[test]
    #[cfg(windows)]
    fn env_map_keys_ci() {
        use ffi::CString;
        use super::EnvKey;
        let mut cmd = Command::new("");
        cmd.env("path", "foo");
        cmd.env("Path", "bar");
        let env = &cmd.env.unwrap();
        let val = env.get(&EnvKey(CString::new(b"PATH").unwrap()));
        assert!(val.unwrap() == &CString::new(b"bar").unwrap());
    }
}
