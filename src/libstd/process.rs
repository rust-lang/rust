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

#![unstable(feature = "process", reason = "recently added via RFC 579")]
#![allow(non_upper_case_globals)]

use prelude::v1::*;
use io::prelude::*;

use ffi::AsOsStr;
use fmt;
use io::{self, Error, ErrorKind};
use path::AsPath;
use libc;
use sync::mpsc::{channel, Receiver};
use sys::pipe2::{self, AnonPipe};
use sys::process2::Process as ProcessImp;
use sys::process2::Command as CommandImp;
use sys::process2::ExitStatus as ExitStatusImp;
use sys_common::{AsInner, AsInnerMut};
use thread::Thread;

/// Representation of a running or exited child process.
///
/// This structure is used to represent and manage child processes. A child
/// process is created via the `Command` struct, which configures the spawning
/// process and can itself be constructed using a builder-style interface.
///
/// # Example
///
/// ```should_fail
/// # #![feature(process)]
///
/// use std::process::Command;
///
/// let output = Command::new("/bin/cat").arg("file.txt").output().unwrap_or_else(|e| {
///     panic!("failed to execute child: {}", e)
/// });
/// let contents = output.stdout;
/// assert!(output.status.success());
/// ```
pub struct Child {
    handle: ProcessImp,

    /// None until wait() or wait_with_output() is called.
    status: Option<ExitStatusImp>,

    /// The handle for writing to the child's stdin, if it has been captured
    pub stdin: Option<ChildStdin>,

    /// The handle for reading from the child's stdout, if it has been captured
    pub stdout: Option<ChildStdout>,

    /// The handle for reading from the child's stderr, if it has been captured
    pub stderr: Option<ChildStderr>,
}

/// A handle to a child procesess's stdin
pub struct ChildStdin {
    inner: AnonPipe
}

impl Write for ChildStdin {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A handle to a child procesess's stdout
pub struct ChildStdout {
    inner: AnonPipe
}

impl Read for ChildStdout {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

/// A handle to a child procesess's stderr
pub struct ChildStderr {
    inner: AnonPipe
}

impl Read for ChildStderr {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

/// The `Command` type acts as a process builder, providing fine-grained control
/// over how a new process should be spawned. A default configuration can be
/// generated using `Command::new(program)`, where `program` gives a path to the
/// program to be executed. Additional builder methods allow the configuration
/// to be changed (for example, by adding arguments) prior to spawning:
///
/// ```
/// # #![feature(process)]
///
/// use std::process::Command;
///
/// let output = Command::new("sh").arg("-c").arg("echo hello").output().unwrap_or_else(|e| {
///   panic!("failed to execute process: {}", e)
/// });
/// let hello = output.stdout;
/// ```
pub struct Command {
    inner: CommandImp,

    // Details explained in the builder methods
    stdin: Option<StdioImp>,
    stdout: Option<StdioImp>,
    stderr: Option<StdioImp>,
}

impl Command {
    /// Constructs a new `Command` for launching the program at
    /// path `program`, with the following default configuration:
    ///
    /// * No arguments to the program
    /// * Inherit the current process's environment
    /// * Inherit the current process's working directory
    /// * Inherit stdin/stdout/stderr for `run` or `status`, but create pipes for `output`
    ///
    /// Builder methods are provided to change these defaults and
    /// otherwise configure the process.
    pub fn new<S: AsOsStr + ?Sized>(program: &S) -> Command {
        Command {
            inner: CommandImp::new(program.as_os_str()),
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    /// Add an argument to pass to the program.
    pub fn arg<S: AsOsStr + ?Sized>(&mut self, arg: &S) -> &mut Command {
        self.inner.arg(arg.as_os_str());
        self
    }

    /// Add multiple arguments to pass to the program.
    pub fn args<S: AsOsStr>(&mut self, args: &[S]) -> &mut Command {
        self.inner.args(args.iter().map(AsOsStr::as_os_str));
        self
    }

    /// Inserts or updates an environment variable mapping.
    ///
    /// Note that environment variable names are case-insensitive (but case-preserving) on Windows,
    /// and case-sensitive on all other platforms.
    pub fn env<S: ?Sized, T: ?Sized>(&mut self, key: &S, val: &T) -> &mut Command where
        S: AsOsStr, T: AsOsStr
    {
        self.inner.env(key.as_os_str(), val.as_os_str());
        self
    }

    /// Removes an environment variable mapping.
    pub fn env_remove<S: ?Sized + AsOsStr>(&mut self, key: &S) -> &mut Command {
        self.inner.env_remove(key.as_os_str());
        self
    }

    /// Clears the entire environment map for the child process.
    pub fn env_clear(&mut self) -> &mut Command {
        self.inner.env_clear();
        self
    }

    /// Set the working directory for the child process.
    pub fn current_dir<P: AsPath + ?Sized>(&mut self, dir: &P) -> &mut Command {
        self.inner.cwd(dir.as_path().as_os_str());
        self
    }

    /// Configuration for the child process's stdin handle (file descriptor 0).
    /// Defaults to `CreatePipe(true, false)` so the input can be written to.
    pub fn stdin(&mut self, cfg: Stdio) -> &mut Command {
        self.stdin = Some(cfg.0);
        self
    }

    /// Configuration for the child process's stdout handle (file descriptor 1).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    pub fn stdout(&mut self, cfg: Stdio) -> &mut Command {
        self.stdout = Some(cfg.0);
        self
    }

    /// Configuration for the child process's stderr handle (file descriptor 2).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    pub fn stderr(&mut self, cfg: Stdio) -> &mut Command {
        self.stderr = Some(cfg.0);
        self
    }

    fn spawn_inner(&self, default_io: StdioImp) -> io::Result<Child> {
        let (their_stdin, our_stdin) = try!(
            setup_io(self.stdin.as_ref().unwrap_or(&default_io), 0, true)
        );
        let (their_stdout, our_stdout) = try!(
            setup_io(self.stdout.as_ref().unwrap_or(&default_io), 1, false)
        );
        let (their_stderr, our_stderr) = try!(
            setup_io(self.stderr.as_ref().unwrap_or(&default_io), 2, false)
        );

        match ProcessImp::spawn(&self.inner, their_stdin, their_stdout, their_stderr) {
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
    /// By default, stdin, stdout and stderr are inherited by the parent.
    pub fn spawn(&mut self) -> io::Result<Child> {
        self.spawn_inner(StdioImp::Inherit)
    }

    /// Executes the command as a child process, waiting for it to finish and
    /// collecting all of its output.
    ///
    /// By default, stdin, stdout and stderr are captured (and used to
    /// provide the resulting output).
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(process)]
    /// use std::process::Command;
    ///
    /// let output = Command::new("cat").arg("foot.txt").output().unwrap_or_else(|e| {
    ///     panic!("failed to execute process: {}", e)
    /// });
    ///
    /// println!("status: {}", output.status);
    /// println!("stdout: {}", String::from_utf8_lossy(output.stdout.as_slice()));
    /// println!("stderr: {}", String::from_utf8_lossy(output.stderr.as_slice()));
    /// ```
    pub fn output(&mut self) -> io::Result<Output> {
        self.spawn_inner(StdioImp::Capture).and_then(|p| p.wait_with_output())
    }

    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its exit status.
    ///
    /// By default, stdin, stdout and stderr are inherited by the parent.
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(process)]
    /// use std::process::Command;
    ///
    /// let status = Command::new("ls").status().unwrap_or_else(|e| {
    ///     panic!("failed to execute process: {}", e)
    /// });
    ///
    /// println!("process exited with: {}", status);
    /// ```
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

impl AsInner<CommandImp> for Command {
    fn as_inner(&self) -> &CommandImp { &self.inner }
}

impl AsInnerMut<CommandImp> for Command {
    fn as_inner_mut(&mut self) -> &mut CommandImp { &mut self.inner }
}

fn setup_io(io: &StdioImp, fd: libc::c_int, readable: bool)
            -> io::Result<(Option<AnonPipe>, Option<AnonPipe>)>
{
    use self::StdioImp::*;
    Ok(match *io {
        Null => {
            (None, None)
        }
        Inherit => {
            (Some(AnonPipe::from_fd(fd)), None)
        }
        Capture => {
            let (reader, writer) = try!(unsafe { pipe2::anon_pipe() });
            if readable {
                (Some(reader), Some(writer))
            } else {
                (Some(writer), Some(reader))
            }
        }
    })
}

/// The output of a finished process.
#[derive(PartialEq, Eq, Clone)]
pub struct Output {
    /// The status (exit code) of the process.
    pub status: ExitStatus,
    /// The data that the process wrote to stdout.
    pub stdout: Vec<u8>,
    /// The data that the process wrote to stderr.
    pub stderr: Vec<u8>,
}

/// Describes what to do with a standard io stream for a child process.
pub struct Stdio(StdioImp);

// The internal enum for stdio setup; see below for descriptions.
#[derive(Clone)]
enum StdioImp {
    Capture,
    Inherit,
    Null,
}

impl Stdio {
    /// A new pipe should be arranged to connect the parent and child processes.
    pub fn capture() -> Stdio { Stdio(StdioImp::Capture) }

    /// The child inherits from the corresponding parent descriptor.
    pub fn inherit() -> Stdio { Stdio(StdioImp::Capture) }

    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    pub fn null() -> Stdio { Stdio(StdioImp::Capture) }
}

/// Describes the result of a process after it has terminated.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(ExitStatusImp);

impl ExitStatus {
    /// Was termination successful? Signal termination not considered a success,
    /// and success is defined as a zero exit status.
    pub fn success(&self) -> bool {
        self.0.success()
    }

    /// Return the exit code of the process, if any.
    ///
    /// On Unix, this will return `None` if the process was terminated
    /// by a signal; `std::os::unix` provides an extension trait for
    /// extracting the signal and other details from the `ExitStatus`.
    pub fn code(&self) -> Option<i32> {
        self.0.code()
    }
}

impl AsInner<ExitStatusImp> for ExitStatus {
    fn as_inner(&self) -> &ExitStatusImp { &self.0 }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Child {
    /// Forces the child to exit. This is equivalent to sending a
    /// SIGKILL on unix platforms.
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
                None
            ))
        }

        unsafe { self.handle.kill() }
    }

    /// Wait for the child to exit completely, returning the status that it
    /// exited with. This function will continue to have the same return value
    /// after it has been called at least once.
    ///
    /// The stdin handle to the child process, if any, will be closed
    /// before waiting. This helps avoid deadlock: it ensures that the
    /// child does not block waiting for input from the parent, while
    /// the parent waits for the child to exit.
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

    /// Simultaneously wait for the child to exit and collect all remaining
    /// output on the stdout/stderr handles, returning a `Output`
    /// instance.
    ///
    /// The stdin handle to the child process, if any, will be closed
    /// before waiting. This helps avoid deadlock: it ensures that the
    /// child does not block waiting for input from the parent, while
    /// the parent waits for the child to exit.
    pub fn wait_with_output(mut self) -> io::Result<Output> {
        drop(self.stdin.take());
        fn read<T: Read + Send>(stream: Option<T>) -> Receiver<io::Result<Vec<u8>>> {
            let (tx, rx) = channel();
            match stream {
                Some(stream) => {
                    Thread::spawn(move || {
                        let mut stream = stream;
                        let mut ret = Vec::new();
                        let res = stream.read_to_end(&mut ret);
                        tx.send(res.map(|_| ret)).unwrap();
                    });
                }
                None => tx.send(Ok(Vec::new())).unwrap()
            }
            rx
        }
        let stdout = read(self.stdout.take());
        let stderr = read(self.stderr.take());
        let status = try!(self.wait());

        Ok(Output {
            status: status,
            stdout: stdout.recv().unwrap().unwrap_or(Vec::new()),
            stderr:  stderr.recv().unwrap().unwrap_or(Vec::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use io::ErrorKind;
    use io::prelude::*;
    use prelude::v1::{Ok, Err, range, drop, Some, None, Vec};
    use prelude::v1::{String, Clone};
    use prelude::v1::{SliceExt, Str, StrExt, AsSlice, ToString, GenericPath};
    use path::Path;
    use old_path;
    use old_io::fs::PathExtensions;
    use rt::running_on_valgrind;
    use str;
    use super::{Child, Command, Output, ExitStatus, Stdio};
    use sync::mpsc::channel;
    use thread::Thread;
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
        assert!(p.wait().unwrap().code() == Some(1));
        drop(p.wait().clone());
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn signal_reported_right() {
        use os::unix::ExitStatusExt;

        let p = Command::new("/bin/sh").arg("-c").arg("kill -1 $$").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        match p.wait().unwrap().signal() {
            Some(1) => {},
            result => panic!("not terminated by signal 1 (instead, {:?})", result),
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
        cmd.arg("foobar").stdout(Stdio::capture());
        assert_eq!(run_output(cmd), "foobar\n");
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn set_current_dir_works() {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg("pwd")
           .current_dir("/")
           .stdout(Stdio::capture());
        assert_eq!(run_output(cmd), "/\n");
    }

    #[cfg(all(unix, not(target_os="android")))]
    #[test]
    fn stdin_works() {
        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("read line; echo $line")
                            .stdin(Stdio::capture())
                            .stdout(Stdio::capture())
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
        use os::unix::*;
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
        use os::unix::*;
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
            Err(e) => assert_eq!(e.kind(), ErrorKind::FileNotFound),
            Ok(..) => panic!()
        }
    }

    #[cfg(not(target_os="android"))]
    #[test]
    fn test_process_output_output() {
        let Output {status, stdout, stderr}
             = Command::new("echo").arg("hello").output().unwrap();
        let output_str = str::from_utf8(stdout.as_slice()).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(stderr, Vec::new());
        }
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
        let prog = Command::new("echo").arg("hello").stdout(Stdio::capture())
            .spawn().unwrap();
        let Output {status, stdout, stderr} = prog.wait_with_output().unwrap();
        let output_str = str::from_utf8(stdout.as_slice()).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello");
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(stderr, Vec::new());
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

        let output = String::from_utf8(prog.wait_with_output().unwrap().stdout).unwrap();
        let parent_dir = os::getcwd().unwrap();
        let child_dir = old_path::Path::new(output.trim());

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
        let result = pwd_cmd().current_dir(&parent_dir).output().unwrap();

        let output = String::from_utf8(result.stdout).unwrap();
        let child_dir = old_path::Path::new(output.trim());

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

        let result = env_cmd().output().unwrap();
        let output = String::from_utf8(result.stdout).unwrap();

        let r = os::env();
        for &(ref k, ref v) in &r {
            // don't check windows magical empty-named variables
            assert!(k.is_empty() ||
                    output.contains(format!("{}={}", *k, *v).as_slice()),
                    "output doesn't contain `{}={}`\n{}",
                    k, v, output);
        }
    }
    #[cfg(target_os="android")]
    #[test]
    fn test_inherit_env() {
        use os;
        if running_on_valgrind() { return; }

        let mut result = env_cmd().output().unwrap();
        let output = String::from_utf8(result.stdout).unwrap();

        let r = os::env();
        for &(ref k, ref v) in &r {
            // don't check android RANDOM variables
            if *k != "RANDOM".to_string() {
                assert!(output.contains(format!("{}={}",
                                                *k,
                                                *v).as_slice()) ||
                        output.contains(format!("{}=\'{}\'",
                                                *k,
                                                *v).as_slice()));
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
        let output = String::from_utf8_lossy(result.stdout.as_slice()).to_string();

        assert!(output.contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    }

    #[test]
    fn test_add_to_env() {
        let result = env_cmd().env("RUN_TEST_NEW_ENV", "123").output().unwrap();
        let output = String::from_utf8_lossy(result.stdout.as_slice()).to_string();

        assert!(output.contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    }
}
