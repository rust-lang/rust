// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Working with processes.

#![stable(feature = "process", since = "1.0.0")]
#![allow(non_upper_case_globals)]

use prelude::v1::*;
use io::prelude::*;

use ffi::OsStr;
use fmt;
use io::{self, Error, ErrorKind};
use path;
use sys::pipe::{self, AnonPipe};
use sys::process as imp;
use sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};
use thread::{self, JoinHandle};

/// Representation of a running or exited child process.
///
/// This structure is used to represent and manage child processes. A child
/// process is created via the `Command` struct, which configures the spawning
/// process and can itself be constructed using a builder-style interface.
///
/// # Examples
///
/// ```should_panic
/// use std::process::Command;
///
/// let mut child = Command::new("/bin/cat")
///                         .arg("file.txt")
///                         .spawn()
///                         .unwrap_or_else(|e| { panic!("failed to execute child: {}", e) });
///
/// let ecode = child.wait()
///                  .unwrap_or_else(|e| { panic!("failed to wait on child: {}", e) });
///
/// assert!(ecode.success());
/// ```
#[stable(feature = "process", since = "1.0.0")]
pub struct Child {
    handle: imp::Process,

    /// None until wait() or wait_with_output() is called.
    status: Option<imp::ExitStatus>,

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

impl IntoInner<imp::Process> for Child {
    fn into_inner(self) -> imp::Process { self.handle }
}

/// A handle to a child process's stdin
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

/// A handle to a child process's stdout
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStdout {
    inner: AnonPipe
}

#[stable(feature = "process", since = "1.0.0")]
impl Read for ChildStdout {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl AsInner<AnonPipe> for ChildStdout {
    fn as_inner(&self) -> &AnonPipe { &self.inner }
}

impl IntoInner<AnonPipe> for ChildStdout {
    fn into_inner(self) -> AnonPipe { self.inner }
}

/// A handle to a child process's stderr
#[stable(feature = "process", since = "1.0.0")]
pub struct ChildStderr {
    inner: AnonPipe
}

#[stable(feature = "process", since = "1.0.0")]
impl Read for ChildStderr {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl AsInner<AnonPipe> for ChildStderr {
    fn as_inner(&self) -> &AnonPipe { &self.inner }
}

impl IntoInner<AnonPipe> for ChildStderr {
    fn into_inner(self) -> AnonPipe { self.inner }
}

/// The `Command` type acts as a process builder, providing fine-grained control
/// over how a new process should be spawned. A default configuration can be
/// generated using `Command::new(program)`, where `program` gives a path to the
/// program to be executed. Additional builder methods allow the configuration
/// to be changed (for example, by adding arguments) prior to spawning:
///
/// ```
/// use std::process::Command;
///
/// let output = Command::new("sh")
///                      .arg("-c")
///                      .arg("echo hello")
///                      .output()
///                      .unwrap_or_else(|e| { panic!("failed to execute process: {}", e) });
/// let hello = output.stdout;
/// ```
#[stable(feature = "process", since = "1.0.0")]
pub struct Command {
    inner: imp::Command,

    // Details explained in the builder methods
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
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
    #[stable(feature = "process", since = "1.0.0")]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Command {
        Command {
            inner: imp::Command::new(program.as_ref()),
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    /// Add an argument to pass to the program.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Command {
        self.inner.arg(arg.as_ref());
        self
    }

    /// Add multiple arguments to pass to the program.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn args<S: AsRef<OsStr>>(&mut self, args: &[S]) -> &mut Command {
        self.inner.args(args.iter().map(AsRef::as_ref));
        self
    }

    /// Inserts or updates an environment variable mapping.
    ///
    /// Note that environment variable names are case-insensitive (but case-preserving) on Windows,
    /// and case-sensitive on all other platforms.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn env<K, V>(&mut self, key: K, val: V) -> &mut Command
        where K: AsRef<OsStr>, V: AsRef<OsStr>
    {
        self.inner.env(key.as_ref(), val.as_ref());
        self
    }

    /// Removes an environment variable mapping.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Command {
        self.inner.env_remove(key.as_ref());
        self
    }

    /// Clears the entire environment map for the child process.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn env_clear(&mut self) -> &mut Command {
        self.inner.env_clear();
        self
    }

    /// Sets the working directory for the child process.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn current_dir<P: AsRef<path::Path>>(&mut self, dir: P) -> &mut Command {
        self.inner.cwd(dir.as_ref().as_ref());
        self
    }

    /// Configuration for the child process's stdin handle (file descriptor 0).
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stdin(&mut self, cfg: Stdio) -> &mut Command {
        self.stdin = Some(cfg);
        self
    }

    /// Configuration for the child process's stdout handle (file descriptor 1).
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stdout(&mut self, cfg: Stdio) -> &mut Command {
        self.stdout = Some(cfg);
        self
    }

    /// Configuration for the child process's stderr handle (file descriptor 2).
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stderr(&mut self, cfg: Stdio) -> &mut Command {
        self.stderr = Some(cfg);
        self
    }

    fn spawn_inner(&self, default_io: StdioImp) -> io::Result<Child> {
        let default_io = Stdio(default_io);

        // See comment on `setup_io` for what `_drop_later` is.
        let (their_stdin, our_stdin, _drop_later) = try!(
            setup_io(self.stdin.as_ref().unwrap_or(&default_io), true)
        );
        let (their_stdout, our_stdout, _drop_later) = try!(
            setup_io(self.stdout.as_ref().unwrap_or(&default_io), false)
        );
        let (their_stderr, our_stderr, _drop_later) = try!(
            setup_io(self.stderr.as_ref().unwrap_or(&default_io), false)
        );

        match imp::Process::spawn(&self.inner, their_stdin, their_stdout,
                                  their_stderr) {
            Err(e) => Err(e),
            Ok(handle) => Ok(Child {
                handle: handle,
                status: None,
                stdin: our_stdin.map(|fd| ChildStdin { inner: fd }),
                stdout: our_stdout.map(|fd| ChildStdout { inner: fd }),
                stderr: our_stderr.map(|fd| ChildStderr { inner: fd }),
            })
        }
    }

    /// Executes the command as a child process, returning a handle to it.
    ///
    /// By default, stdin, stdout and stderr are inherited from the parent.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn spawn(&mut self) -> io::Result<Child> {
        self.spawn_inner(StdioImp::Inherit)
    }

    /// Executes the command as a child process, waiting for it to finish and
    /// collecting all of its output.
    ///
    /// By default, stdin, stdout and stderr are captured (and used to
    /// provide the resulting output).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::process::Command;
    /// let output = Command::new("cat").arg("foo.txt").output().unwrap_or_else(|e| {
    ///     panic!("failed to execute process: {}", e)
    /// });
    ///
    /// println!("status: {}", output.status);
    /// println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    /// println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn output(&mut self) -> io::Result<Output> {
        self.spawn_inner(StdioImp::MakePipe).and_then(|p| p.wait_with_output())
    }

    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its exit status.
    ///
    /// By default, stdin, stdout and stderr are inherited from the parent.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::process::Command;
    ///
    /// let status = Command::new("ls").status().unwrap_or_else(|e| {
    ///     panic!("failed to execute process: {}", e)
    /// });
    ///
    /// println!("process exited with: {}", status);
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn status(&mut self) -> io::Result<ExitStatus> {
        self.spawn().and_then(|mut p| p.wait())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Command {
    /// Format the program and arguments of a Command for display. Any
    /// non-utf8 data is lossily converted using the utf8 replacement
    /// character.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{:?}", self.inner.program));
        for arg in &self.inner.args {
            try!(write!(f, " {:?}", arg));
        }
        Ok(())
    }
}

impl AsInner<imp::Command> for Command {
    fn as_inner(&self) -> &imp::Command { &self.inner }
}

impl AsInnerMut<imp::Command> for Command {
    fn as_inner_mut(&mut self) -> &mut imp::Command { &mut self.inner }
}

// Takes a `Stdio` configuration (this module) and whether the to-be-owned
// handle will be readable.
//
// Returns a triple of (stdio to spawn with, stdio to store, stdio to drop). The
// stdio to spawn with is passed down to the `sys` module and indicates how the
// stdio stream should be set up. The "stdio to store" is an object which
// should be returned in the `Child` that makes its way out. The "stdio to drop"
// represents the raw value of "stdio to spawn with", but is the owned variant
// for it. This needs to be dropped after the child spawns
fn setup_io(io: &Stdio, readable: bool)
            -> io::Result<(imp::Stdio, Option<AnonPipe>, Option<AnonPipe>)>
{
    Ok(match io.0 {
        StdioImp::MakePipe => {
            let (reader, writer) = try!(pipe::anon_pipe());
            if readable {
                (imp::Stdio::Raw(reader.raw()), Some(writer), Some(reader))
            } else {
                (imp::Stdio::Raw(writer.raw()), Some(reader), Some(writer))
            }
        }
        StdioImp::Raw(ref owned) => (imp::Stdio::Raw(owned.raw()), None, None),
        StdioImp::Inherit => (imp::Stdio::Inherit, None, None),
        StdioImp::None => (imp::Stdio::None, None, None),
    })
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

/// Describes what to do with a standard I/O stream for a child process.
#[stable(feature = "process", since = "1.0.0")]
pub struct Stdio(StdioImp);

// The internal enum for stdio setup; see below for descriptions.
enum StdioImp {
    MakePipe,
    Raw(imp::RawStdio),
    Inherit,
    None,
}

impl Stdio {
    /// A new pipe should be arranged to connect the parent and child processes.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn piped() -> Stdio { Stdio(StdioImp::MakePipe) }

    /// The child inherits from the corresponding parent descriptor.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn inherit() -> Stdio { Stdio(StdioImp::Inherit) }

    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    #[stable(feature = "process", since = "1.0.0")]
    pub fn null() -> Stdio { Stdio(StdioImp::None) }
}

impl FromInner<imp::RawStdio> for Stdio {
    fn from_inner(inner: imp::RawStdio) -> Stdio {
        Stdio(StdioImp::Raw(inner))
    }
}

/// Describes the result of a process after it has terminated.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "process", since = "1.0.0")]
pub struct ExitStatus(imp::ExitStatus);

impl ExitStatus {
    /// Was termination successful? Signal termination not considered a success,
    /// and success is defined as a zero exit status.
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

#[stable(feature = "process", since = "1.0.0")]
impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Child {
    /// Forces the child to exit. This is equivalent to sending a
    /// SIGKILL on unix platforms.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn kill(&mut self) -> io::Result<()> {
        #[cfg(unix)] fn collect_status(p: &mut Child) {
            // On Linux (and possibly other unices), a process that has exited will
            // continue to accept signals because it is "defunct". The delivery of
            // signals will only fail once the child has been reaped. For this
            // reason, if the process hasn't exited yet, then we attempt to collect
            // their status with WNOHANG.
            if p.status.is_none() {
                match p.handle.try_wait() {
                    Some(status) => { p.status = Some(status); }
                    None => {}
                }
            }
        }
        #[cfg(windows)] fn collect_status(_p: &mut Child) {}

        collect_status(self);

        // if the process has finished, and therefore had waitpid called,
        // and we kill it, then on unix we might ending up killing a
        // newer process that happens to have the same (re-used) id
        if self.status.is_some() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "invalid argument: can't kill an exited process",
            ))
        }

        unsafe { self.handle.kill() }
    }

    /// Returns the OS-assigned process identifier associated with this child.
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
    #[stable(feature = "process", since = "1.0.0")]
    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        drop(self.stdin.take());
        match self.status {
            Some(code) => Ok(ExitStatus(code)),
            None => {
                let status = try!(self.handle.wait());
                self.status = Some(status);
                Ok(ExitStatus(status))
            }
        }
    }

    /// Simultaneously waits for the child to exit and collect all remaining
    /// output on the stdout/stderr handles, returning a `Output`
    /// instance.
    ///
    /// The stdin handle to the child process, if any, will be closed
    /// before waiting. This helps avoid deadlock: it ensures that the
    /// child does not block waiting for input from the parent, while
    /// the parent waits for the child to exit.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn wait_with_output(mut self) -> io::Result<Output> {
        drop(self.stdin.take());
        fn read<R>(mut input: R) -> JoinHandle<io::Result<Vec<u8>>>
            where R: Read + Send + 'static
        {
            thread::spawn(move || {
                let mut ret = Vec::new();
                input.read_to_end(&mut ret).map(|_| ret)
            })
        }
        let stdout = self.stdout.take().map(read);
        let stderr = self.stderr.take().map(read);
        let status = try!(self.wait());
        let stdout = stdout.and_then(|t| t.join().unwrap().ok());
        let stderr = stderr.and_then(|t| t.join().unwrap().ok());

        Ok(Output {
            status: status,
            stdout: stdout.unwrap_or(Vec::new()),
            stderr: stderr.unwrap_or(Vec::new()),
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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn exit(code: i32) -> ! {
    ::sys_common::cleanup();
    ::sys::os::exit(code)
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use io::prelude::*;

    use io::ErrorKind;
    use str;
    use super::{Command, Output, Stdio};

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
        assert!(p.wait().unwrap().code() == Some(1));
        drop(p.wait());
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
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

    #[cfg(not(target_os="android"))]
    #[test]
    fn stdout_works() {
        let mut cmd = Command::new("echo");
        cmd.arg("foobar").stdout(Stdio::piped());
        assert_eq!(run_output(cmd), "foobar\n");
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn set_current_dir_works() {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg("pwd")
           .current_dir("/")
           .stdout(Stdio::piped());
        assert_eq!(run_output(cmd), "/\n");
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
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


    #[cfg(all(unix, not(target_os="android")))]
    #[test]
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

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn uid_to_root_fails() {
        use os::unix::prelude::*;
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
        assert!(status.code() == Some(1));

        status = Command::new("true").status().unwrap();
        assert!(status.success());
    }

    #[test]
    fn test_process_output_fail_to_start() {
        match Command::new("/no-binary-by-this-name-should-exist").output() {
            Err(e) => assert_eq!(e.kind(), ErrorKind::NotFound),
            Ok(..) => panic!()
        }
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_process_output_output() {
        let Output {status, stdout, stderr}
             = Command::new("echo").arg("hello").output().unwrap();
        let output_str = str::from_utf8(&stdout).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello");
        assert_eq!(stderr, Vec::new());
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_process_output_error() {
        let Output {status, stdout, stderr}
             = Command::new("mkdir").arg(".").output().unwrap();

        assert!(status.code() == Some(1));
        assert_eq!(stdout, Vec::new());
        assert!(!stderr.is_empty());
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_finish_once() {
        let mut prog = Command::new("false").spawn().unwrap();
        assert!(prog.wait().unwrap().code() == Some(1));
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_finish_twice() {
        let mut prog = Command::new("false").spawn().unwrap();
        assert!(prog.wait().unwrap().code() == Some(1));
        assert!(prog.wait().unwrap().code() == Some(1));
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_wait_with_output_once() {
        let prog = Command::new("echo").arg("hello").stdout(Stdio::piped())
            .spawn().unwrap();
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

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_inherit_env() {
        use env;

        let result = env_cmd().output().unwrap();
        let output = String::from_utf8(result.stdout).unwrap();

        for (ref k, ref v) in env::vars() {
            // Windows has hidden environment variables whose names start with
            // equals signs (`=`). Those do not show up in the output of the
            // `set` command.
            assert!((cfg!(windows) && k.starts_with("=")) ||
                    output.contains(&format!("{}={}", *k, *v)),
                    "output doesn't contain `{}={}`\n{}",
                    k, v, output);
        }
    }
    #[cfg(target_os="android")]
    #[test]
    fn test_inherit_env() {
        use env;

        let mut result = env_cmd().output().unwrap();
        let output = String::from_utf8(result.stdout).unwrap();

        for (ref k, ref v) in env::vars() {
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
}
