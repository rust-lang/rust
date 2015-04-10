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
use fs::File;
use io::{self, Error, ErrorKind};
use libc;
use path;
use sync::mpsc::{channel, Receiver};
use sys::pipe2::{self, AnonPipe};
use sys::process2::Command as CommandImp;
use sys::process2::Process as ProcessImp;
use sys::process2::ExitStatus as ExitStatusImp;
use sys_common::{AsInner, AsInnerMut};
use thread;

/// Representation of a running or exited child process.
///
/// This structure is used to represent and manage child processes. A child
/// process is created via the `Command` struct, which configures the spawning
/// process and can itself be constructed using a builder-style interface.
///
/// # Examples
///
/// ```should_panic
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
#[stable(feature = "process", since = "1.0.0")]
pub struct Child {
    handle: ProcessImp,

    /// None until wait() or wait_with_output() is called.
    status: Option<ExitStatusImp>,

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

/// A handle to a child procesess's stdin
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

/// A handle to a child procesess's stdout
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

/// A handle to a child procesess's stderr
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

/// The `Command` type acts as a process builder, providing fine-grained control
/// over how a new process should be spawned. A default configuration can be
/// generated using `Command::new(program)`, where `program` gives a path to the
/// program to be executed. Additional builder methods allow the configuration
/// to be changed (for example, by adding arguments) prior to spawning:
///
/// ```
/// use std::process::Command;
///
/// let output = Command::new("sh").arg("-c").arg("echo hello").output().unwrap_or_else(|e| {
///   panic!("failed to execute process: {}", e)
/// });
/// let hello = output.stdout;
/// ```
#[stable(feature = "process", since = "1.0.0")]
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
    #[stable(feature = "process", since = "1.0.0")]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Command {
        Command {
            inner: CommandImp::new(program.as_ref()),
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

    /// Set the working directory for the child process.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn current_dir<P: AsRef<path::Path>>(&mut self, dir: P) -> &mut Command {
        self.inner.cwd(dir.as_ref().as_ref());
        self
    }

    /// Configuration for the child process's stdin handle (file descriptor 0).
    /// Defaults to `CreatePipe(true, false)` so the input can be written to.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stdin(&mut self, cfg: Stdio) -> &mut Command {
        self.stdin = Some(cfg.0);
        self
    }

    /// Configuration for the child process's stdout handle (file descriptor 1).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn stdout(&mut self, cfg: Stdio) -> &mut Command {
        self.stdout = Some(cfg.0);
        self
    }

    /// Configuration for the child process's stderr handle (file descriptor 2).
    /// Defaults to `CreatePipe(false, true)` so the output can be collected.
    #[stable(feature = "process", since = "1.0.0")]
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
    /// # #![feature(process)]
    /// use std::process::Command;
    ///
    /// let output = Command::new("cat").arg("foot.txt").output().unwrap_or_else(|e| {
    ///     panic!("failed to execute process: {}", e)
    /// });
    ///
    /// println!("status: {}", output.status);
    /// println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    /// println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    /// ```
    #[stable(feature = "process", since = "1.0.0")]
    pub fn output(&mut self) -> io::Result<Output> {
        self.spawn_inner(StdioImp::Piped).and_then(|p| p.wait_with_output())
    }

    /// Executes a command as a child process, waiting for it to finish and
    /// collecting its exit status.
    ///
    /// By default, stdin, stdout and stderr are inherited by the parent.
    ///
    /// # Examples
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
        Piped => {
            let (reader, writer) = try!(unsafe { pipe2::anon_pipe() });
            if readable {
                (Some(reader), Some(writer))
            } else {
                (Some(writer), Some(reader))
            }
        }
        Redirect(ref pipe) => {
            (Some(pipe.clone()), None)
        }
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

/// Describes what to do with a standard io stream for a child process.
#[stable(feature = "process", since = "1.0.0")]
pub struct Stdio(StdioImp);

// The internal enum for stdio setup; see below for descriptions.
#[derive(Clone)]
enum StdioImp {
    Piped,
    Inherit,
    Null,
    Redirect(AnonPipe),
}

impl Stdio {
    /// A new pipe should be arranged to connect the parent and child processes.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn piped() -> Stdio { Stdio(StdioImp::Piped) }

    /// The child inherits from the corresponding parent descriptor.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn inherit() -> Stdio { Stdio(StdioImp::Inherit) }

    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    #[stable(feature = "process", since = "1.0.0")]
    pub fn null() -> Stdio { Stdio(StdioImp::Null) }

    /// This stream will use a currently opened file. The file must have been
    /// opened with the appropriate read/write permissions.
    ///
    /// Note: this call will make an internal duplicate of the file handle, so
    /// when using a file handle created as part of a pipe (e.g. from `std::fs::Pipe`)
    /// keep in mind that an extra copy of it will remain open until it is dropped.
    /// Thus it is possible to deadlock when reading from the pipe if writers to that
    /// pipe remain open, e.g. the original argument to this method and its returned value.
    ///
    /// ```
    /// #![feature(fs_pipe)]
    /// #![feature(process_redirect)]
    /// use std::process::{Command, Stdio};
    /// use std::fs::Pipe;
    /// use std::io::Write;
    ///
    /// let pipe = Pipe::new().unwrap_or_else(|e| {
    ///     panic!("unable to create a pipe: {}", e)
    /// });
    ///
    /// let mut write_to_cmd = pipe.writer;
    /// let mut cmd_stdin = pipe.reader;
    ///
    /// let mut cmd = Command::new("cat");
    /// cmd.stdout(Stdio::piped());
    /// cmd.stdin(Stdio::redirect(&mut cmd_stdin, false).unwrap_or_else(|e| {
    ///     panic!("unable to redirect stdin: {}", e);
    /// }));
    ///
    /// let child = cmd.spawn().unwrap_or_else(|e| {
    ///     panic!("failed to spawn process: {}", e);
    /// });
    ///
    /// // This indirectly drops the extra writer copy of
    /// // the pipe, only write_to_cmd remains.
    /// drop(cmd);
    ///
    /// write_to_cmd.write_all("hello world!".as_bytes()).unwrap();
    /// write_to_cmd.flush().unwrap();
    /// drop(write_to_cmd); // Nothing else to write, signal that we are done!
    ///
    /// let output = child.wait_with_output().unwrap_or_else(|e| {
    ///     panic!("failed to get child's output: {}", e);
    /// });
    ///
    /// assert_eq!("hello world!", String::from_utf8(output.stdout).unwrap());
    /// ```
    #[unstable(feature = "process_redirect", reason = "feature was recently added")]
    pub fn redirect(f: &mut File, writable: bool) -> io::Result<Stdio> {
        Ok(Stdio(StdioImp::Redirect(try!(f.as_inner_mut().dup_as_anon_pipe(writable)))))
    }
}

/// Describes the result of a process after it has terminated.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[stable(feature = "process", since = "1.0.0")]
pub struct ExitStatus(ExitStatusImp);

impl ExitStatus {
    /// Was termination successful? Signal termination not considered a success,
    /// and success is defined as a zero exit status.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn success(&self) -> bool {
        self.0.success()
    }

    /// Return the exit code of the process, if any.
    ///
    /// On Unix, this will return `None` if the process was terminated
    /// by a signal; `std::os::unix` provides an extension trait for
    /// extracting the signal and other details from the `ExitStatus`.
    #[stable(feature = "process", since = "1.0.0")]
    pub fn code(&self) -> Option<i32> {
        self.0.code()
    }
}

impl AsInner<ExitStatusImp> for ExitStatus {
    fn as_inner(&self) -> &ExitStatusImp { &self.0 }
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

    /// Wait for the child to exit completely, returning the status that it
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

    /// Simultaneously wait for the child to exit and collect all remaining
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
        fn read<T: Read + Send + 'static>(stream: Option<T>) -> Receiver<io::Result<Vec<u8>>> {
            let (tx, rx) = channel();
            match stream {
                Some(stream) => {
                    thread::spawn(move || {
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
    ::sys::os::exit(code)
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use io::prelude::*;

    use fs::{File, OpenOptions, Pipe, remove_file};
    use io::ErrorKind;
    use old_path::{self, GenericPath};
    use old_io::fs::PathExtensions;
    use rt::running_on_valgrind;
    use str;
    use super::{Command, Output, Stdio};

    // Poor man's mktemp
    macro_rules! unique_test_path {
        () => {
            {
                use env;
                use path::Path;
                let name = Path::new(file!()).file_name().unwrap().to_str().unwrap();
                let mut path = env::temp_dir();
                path.set_file_name(format!("rust-test-{}:{}", name, line!()));
                path
            }
        };
    }

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

        let p = Command::new("/bin/sh").arg("-c").arg("kill -9 $$").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        match p.wait().unwrap().signal() {
            Some(9) => {},
            result => panic!("not terminated by signal 9 (instead, {:?})", result),
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
        let prog = Command::new("echo").arg("hello").stdout(Stdio::piped())
            .spawn().unwrap();
        let Output {status, stdout, stderr} = prog.wait_with_output().unwrap();
        let output_str = str::from_utf8(&stdout).unwrap();

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

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn test_keep_current_working_dir() {
        use os;
        let prog = pwd_cmd().spawn().unwrap();

        let output = String::from_utf8(prog.wait_with_output().unwrap().stdout).unwrap();
        let parent_dir = ::env::current_dir().unwrap().to_str().unwrap().to_string();
        let parent_dir = old_path::Path::new(parent_dir);
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
        let parent_dir = ::env::current_dir().unwrap().to_str().unwrap().to_string();
        let parent_dir = old_path::Path::new(parent_dir).dir_path();
        let result = pwd_cmd().current_dir(parent_dir.as_str().unwrap()).output().unwrap();

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
        use std::env;
        if running_on_valgrind() { return; }

        let result = env_cmd().output().unwrap();
        let output = String::from_utf8(result.stdout).unwrap();

        for (ref k, ref v) in env::vars() {
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
        use std::env;
        if running_on_valgrind() { return; }

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

    #[cfg(not(target_os="android"))]
    fn shell_cmd(c_flag: bool) -> Command {
        let mut cmd = Command::new("sh");
        if c_flag { cmd.arg("-c"); }
        cmd
    }

    #[cfg(target_os="android")]
    fn shell_cmd(c_flag: bool) -> Command {
        let mut cmd = Command::new("/system/bin/sh");
        if c_flag { cmd.arg("-c"); }
        cmd
    }

    #[test]
    fn test_redirect_stdio_with_file() {
        let in_path = unique_test_path!();
        let out_path = unique_test_path!();
        let err_path = unique_test_path!();

        {
            let in_path = in_path.clone();
            let mut in_file = File::create(in_path).ok().expect("failed to open stdin file");
            in_file.write_all("echo stdout\necho stderr 1>&2\n".as_bytes()).unwrap();
            in_file.flush().unwrap();
        }

        {
            let in_path = in_path.clone();
            let out_path = out_path.clone();
            let err_path = err_path.clone();

            let mut in_file = File::open(in_path).ok().expect("failed to open stdout file");
            let mut out_file = File::create(out_path).ok().expect("failed to open stdout file");
            let mut err_file = File::create(err_path).ok().expect("failed to open stderr file");

            let mut cmd = shell_cmd(false);
            cmd.stdin(Stdio::redirect(&mut in_file, false).unwrap());
            cmd.stdout(Stdio::redirect(&mut out_file, true).unwrap());
            cmd.stderr(Stdio::redirect(&mut err_file, true).unwrap());

            assert!(cmd.status().ok().expect("unabled to spawn child").success());
        }

        let mut out_file = File::open(out_path.clone()).ok().expect("missing stdout file");
        let mut err_file = File::open(err_path.clone()).ok().expect("missing stderr file");

        let mut stdout = String::new();
        let mut stderr = String::new();

        out_file.read_to_string(&mut stdout).ok().expect("failed to read from stdout file");
        err_file.read_to_string(&mut stderr).ok().expect("failed to read from stderr file");

        assert_eq!(stdout, "stdout\n");
        assert_eq!(stderr, "stderr\n");

        let _ = remove_file(in_path);
        let _ = remove_file(out_path);
        let _ = remove_file(err_path);
    }

    #[test]
    fn test_redirect_reuse() {
        let out_path = unique_test_path!();

        {
            let out_path = out_path.clone();
            let mut out_file = File::create(out_path).ok().expect("failed to open stdout file");

            let mut cmd = shell_cmd(true);
            cmd.arg("echo stdout\necho stderr 1>&2\n");
            cmd.stdout(Stdio::redirect(&mut out_file, true).unwrap());
            cmd.stderr(Stdio::redirect(&mut out_file, true).unwrap());

            assert!(cmd.status().ok().expect("unabled to spawn child").success());
        }

        let mut out_file = File::open(out_path.clone()).ok().expect("missing stdout file");

        let mut stdout = String::new();
        out_file.read_to_string(&mut stdout).ok().expect("failed to read from stdout file");
        assert_eq!(stdout, "stdout\nstderr\n");

        let _ = remove_file(out_path);
    }

    #[test]
    fn test_redirect_reuse_as_stdin_and_stdout() {
        use io::SeekFrom;

        let path = unique_test_path!();
        let mut file = OpenOptions::new().create(true).truncate(true)
            .read(true).write(true).open(path.clone()).ok().expect("failed to open file");

        let mut cmd_writer = shell_cmd(true);
        cmd_writer.arg("echo 'echo stdout\necho stderr 1>&2\n'");
        cmd_writer.stdout(Stdio::redirect(&mut file, true).unwrap());

        let mut cmd_reader = shell_cmd(false);
        cmd_reader.stdin(Stdio::redirect(&mut file, false).unwrap());

        assert!(cmd_writer.status().ok().expect("unabled to spawn writer").success());

        file.seek(SeekFrom::Start(0)).unwrap();

        let output = cmd_reader.output().ok().expect("unable to spawn reader");
        assert_eq!(output.stdout, "stdout\n".as_bytes());
        assert_eq!(output.stderr, "stderr\n".as_bytes());

        let _ = remove_file(path);
    }

    #[test]
    fn test_redirect_file_remains_valid() {
        let out_path = unique_test_path!();
        let mut out_file = File::create(out_path.clone())
            .ok().expect("failed to open stdout file for writing");
        out_file.write_all("pre-cmd write\n".as_bytes()).unwrap();
        out_file.flush().unwrap();

        {
            let mut cmd = shell_cmd(true);
            cmd.arg("echo stdout\necho stderr 1>&2\n");
            cmd.stdout(Stdio::redirect(&mut out_file, true).unwrap());
            cmd.stderr(Stdio::redirect(&mut out_file, true).unwrap());

            assert!(cmd.status().ok().expect("unabled to spawn child").success());
        }

        out_file.write_all("post-cmd write\n".as_bytes()).unwrap();
        out_file.flush().unwrap();

        let mut out_read = File::open(out_path.clone())
            .ok().expect("failed to open stdout for reading");

        let mut stdout = String::new();
        out_read.read_to_string(&mut stdout).ok().expect("failed to read from stdout file");
        assert_eq!(stdout, "pre-cmd write\nstdout\nstderr\npost-cmd write\n");

        let _ = remove_file(out_path);
    }

    #[test]
    fn test_redirect_stdio_with_pipe() {
        let (mut cmd_stdin, mut send_to_cmd) = {
            let pipe = Pipe::new().ok().expect("unable to create pipe");
            (pipe.reader, pipe.writer)
        };

        let (mut read_from_cmd, mut cmd_stdout) = {
            let pipe = Pipe::new().ok().expect("unable to create pipe");
            (pipe.reader, pipe.writer)
        };

        {
            let mut cmd = shell_cmd(false);
            cmd.stdin(Stdio::redirect(&mut cmd_stdin, false).unwrap());
            cmd.stdout(Stdio::redirect(&mut cmd_stdout, true).unwrap());
            cmd.stderr(Stdio::redirect(&mut cmd_stdout, true).unwrap());

            // cmd has a copy of cmd_stdout (the pipe writer), since we don't intend
            // to write anything in this end, we had better close it to avoid hanging
            // while trying to read from the other end.
            drop(cmd_stdout);
            drop(cmd_stdin); // We also don't care about reading from the child's pipe

            let mut child = cmd.spawn().ok().expect("unsable to spawn child");

            send_to_cmd.write("echo stdout\necho stderr 1>&2\nexit 0\n".as_bytes()).unwrap();
            drop(send_to_cmd);

            assert!(child.wait().ok().expect("unable to wait on child").success());

            // cmd goes out of scope here and closes its (duplicated) end of the pipe
        }

        let mut output = String::new();
        read_from_cmd.read_to_string(&mut output).ok().expect("failed to read from stdout file");
        assert_eq!(output, "stdout\nstderr\n");
    }

    #[test]
    fn test_redirect_pipe_remains_valid() {
        let (mut cmd_stdin, mut send_to_cmd) = {
            let pipe = Pipe::new().ok().expect("unable to create pipe");
            (pipe.reader, pipe.writer)
        };

        let (mut read_from_cmd, mut cmd_stdout) = {
            let pipe = Pipe::new().ok().expect("unable to create pipe");
            (pipe.reader, pipe.writer)
        };

        {
            let mut cmd = shell_cmd(false);
            cmd.stdin(Stdio::redirect(&mut cmd_stdin, false).unwrap());
            cmd.stdout(Stdio::redirect(&mut cmd_stdout, true).unwrap());
            cmd.stderr(Stdio::redirect(&mut cmd_stdout, true).unwrap());

            // cmd has a copy of cmd_stdout (the pipe writer), since we don't intend
            // to write anything in this end, we had better close it to avoid hanging
            // while trying to read from the other end.
            drop(cmd_stdout);

            let mut child = cmd.spawn().ok().expect("unsable to spawn child");

            send_to_cmd.write("echo stdout\necho stderr 1>&2\nexit 0\n".as_bytes()).unwrap();
            assert!(child.wait().ok().expect("unable to wait on child").success());

            // cmd goes out of scope here and closes its (duplicated) end of the pipe
        }

        send_to_cmd.write("post cmd\n".as_bytes()).unwrap();

        // Make sure there are no writers left before we read from the pipe
        drop(send_to_cmd);

        let mut output = String::new();
        read_from_cmd.read_to_string(&mut output).unwrap();
        assert_eq!(output, "stdout\nstderr\n");

        let mut output = String::new();
        cmd_stdin.read_to_string(&mut output).unwrap();
        assert_eq!(output, "post cmd\n");
    }
}
