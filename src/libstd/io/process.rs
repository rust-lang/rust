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

use prelude::*;

use str;
use fmt;
use io::{IoResult, IoError};
use io;
use libc;
use mem;
use owned::Box;
use rt::rtio::{RtioProcess, ProcessConfig, IoFactory, LocalIo};
use rt::rtio;
use c_str::CString;

/// Signal a process to exit, without forcibly killing it. Corresponds to
/// SIGTERM on unix platforms.
#[cfg(windows)] pub static PleaseExitSignal: int = 15;
/// Signal a process to exit immediately, forcibly killing it. Corresponds to
/// SIGKILL on unix platforms.
#[cfg(windows)] pub static MustDieSignal: int = 9;
/// Signal a process to exit, without forcibly killing it. Corresponds to
/// SIGTERM on unix platforms.
#[cfg(not(windows))] pub static PleaseExitSignal: int = libc::SIGTERM as int;
/// Signal a process to exit immediately, forcibly killing it. Corresponds to
/// SIGKILL on unix platforms.
#[cfg(not(windows))] pub static MustDieSignal: int = libc::SIGKILL as int;

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
///     Err(e) => fail!("failed to execute child: {}", e),
/// };
///
/// let contents = child.stdout.get_mut_ref().read_to_end();
/// assert!(child.wait().unwrap().success());
/// ```
pub struct Process {
    handle: Box<RtioProcess:Send>,

    /// Handle to the child's stdin, if the `stdin` field of this process's
    /// `ProcessConfig` was `CreatePipe`. By default, this handle is `Some`.
    pub stdin: Option<io::PipeStream>,

    /// Handle to the child's stdout, if the `stdout` field of this process's
    /// `ProcessConfig` was `CreatePipe`. By default, this handle is `Some`.
    pub stdout: Option<io::PipeStream>,

    /// Handle to the child's stderr, if the `stderr` field of this process's
    /// `ProcessConfig` was `CreatePipe`. By default, this handle is `Some`.
    pub stderr: Option<io::PipeStream>,

    /// Extra I/O handles as configured by the original `ProcessConfig` when
    /// this process was created. This is by default empty.
    pub extra_io: Vec<Option<io::PipeStream>>,
}

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
///   Err(e) => fail!("failed to execute process: {}", e),
/// };
///
/// let output = process.stdout.get_mut_ref().read_to_end();
/// ```
pub struct Command {
    // The internal data for the builder. Documented by the builder
    // methods below, and serialized into rt::rtio::ProcessConfig.
    program: CString,
    args: Vec<CString>,
    env: Option<Vec<(CString, CString)>>,
    cwd: Option<CString>,
    stdin: StdioContainer,
    stdout: StdioContainer,
    stderr: StdioContainer,
    extra_io: Vec<StdioContainer>,
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
    pub fn new<T:ToCStr>(program: T) -> Command {
        Command {
            program: program.to_c_str(),
            args: Vec::new(),
            env: None,
            cwd: None,
            stdin: CreatePipe(true, false),
            stdout: CreatePipe(false, true),
            stderr: CreatePipe(false, true),
            extra_io: Vec::new(),
            uid: None,
            gid: None,
            detach: false,
        }
    }

    /// Add an argument to pass to the program.
    pub fn arg<'a, T:ToCStr>(&'a mut self, arg: T) -> &'a mut Command {
        self.args.push(arg.to_c_str());
        self
    }

    /// Add multiple arguments to pass to the program.
    pub fn args<'a, T:ToCStr>(&'a mut self, args: &[T]) -> &'a mut Command {
        self.args.extend(args.iter().map(|arg| arg.to_c_str()));;
        self
    }

    /// Sets the environment for the child process (rather than inheriting it
    /// from the current process).

    // FIXME (#13851): We should change this interface to allow clients to (1)
    // build up the env vector incrementally and (2) allow both inheriting the
    // current process's environment AND overriding/adding additional
    // environment variables. The underlying syscalls assume that the
    // environment has no duplicate names, so we really want to use a hashtable
    // to compute the environment to pass down to the syscall; resolving issue
    // #13851 will make it possible to use the standard hashtable.
    pub fn env<'a, T:ToCStr>(&'a mut self, env: &[(T,T)]) -> &'a mut Command {
        self.env = Some(env.iter().map(|&(ref name, ref val)| {
            (name.to_c_str(), val.to_c_str())
        }).collect());
        self
    }

    /// Set the working directory for the child process.
    pub fn cwd<'a>(&'a mut self, dir: &Path) -> &'a mut Command {
        self.cwd = Some(dir.to_c_str());
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
    /// Attaches a stream/file descriptor/pipe to the child process. Inherited
    /// file descriptors are numbered consecutively, starting at 3; the first
    /// three file descriptors (stdin/stdout/stderr) are configured with the
    /// `stdin`, `stdout`, and `stderr` methods.
    pub fn extra_io<'a>(&'a mut self, cfg: StdioContainer) -> &'a mut Command {
        self.extra_io.push(cfg);
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
        fn to_rtio(p: StdioContainer) -> rtio::StdioContainer {
            match p {
                Ignored => rtio::Ignored,
                InheritFd(fd) => rtio::InheritFd(fd),
                CreatePipe(a, b) => rtio::CreatePipe(a, b),
            }
        }
        let extra_io: Vec<rtio::StdioContainer> =
            self.extra_io.iter().map(|x| to_rtio(*x)).collect();
        LocalIo::maybe_raise(|io| {
            let cfg = ProcessConfig {
                program: &self.program,
                args: self.args.as_slice(),
                env: self.env.as_ref().map(|env| env.as_slice()),
                cwd: self.cwd.as_ref(),
                stdin: to_rtio(self.stdin),
                stdout: to_rtio(self.stdout),
                stderr: to_rtio(self.stderr),
                extra_io: extra_io.as_slice(),
                uid: self.uid,
                gid: self.gid,
                detach: self.detach,
            };
            io.spawn(cfg).map(|(p, io)| {
                let mut io = io.move_iter().map(|p| {
                    p.map(|p| io::PipeStream::new(p))
                });
                Process {
                    handle: p,
                    stdin: io.next().unwrap(),
                    stdout: io.next().unwrap(),
                    stderr: io.next().unwrap(),
                    extra_io: io.collect(),
                }
            })
        }).map_err(IoError::from_rtio_error)
    }

    /// Executes the command as a child process, waiting for it to finish and
    /// collecting all of its output.
    ///
    /// # Example
    ///
    /// ```
    /// use std::io::Command;
    /// use std::str;
    ///
    /// let output = match Command::new("cat").arg("foot.txt").output() {
    ///     Ok(output) => output,
    ///     Err(e) => fail!("failed to execute process: {}", e),
    /// };
    ///
    /// println!("status: {}", output.status);
    /// println!("stdout: {}", str::from_utf8_lossy(output.output.as_slice()));
    /// println!("stderr: {}", str::from_utf8_lossy(output.error.as_slice()));
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
    /// use std::io::Command;
    ///
    /// let status = match Command::new("ls").status() {
    ///     Ok(status) => status,
    ///     Err(e) => fail!("failed to execute process: {}", e),
    /// };
    ///
    /// println!("process exited with: {}", status);
    /// ```
    pub fn status(&self) -> IoResult<ProcessExit> {
        self.spawn().and_then(|mut p| p.wait())
    }
}

impl fmt::Show for Command {
    /// Format the program and arguments of a Command for display. Any
    /// non-utf8 data is lossily converted using the utf8 replacement
    /// character.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{}", str::from_utf8_lossy(self.program.as_bytes_no_nul())));
        for arg in self.args.iter() {
            try!(write!(f, " '{}'", str::from_utf8_lossy(arg.as_bytes_no_nul())));
        }
        Ok(())
    }
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
pub enum StdioContainer {
    /// This stream will be ignored. This is the equivalent of attaching the
    /// stream to `/dev/null`
    Ignored,

    /// The specified file descriptor is inherited for the stream which it is
    /// specified for.
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
        LocalIo::maybe_raise(|io| {
            io.kill(id, signal)
        }).map_err(IoError::from_rtio_error)
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
        self.handle.kill(signal).map_err(IoError::from_rtio_error)
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
        match self.handle.wait() {
            Ok(rtio::ExitSignal(s)) => Ok(ExitSignal(s)),
            Ok(rtio::ExitStatus(s)) => Ok(ExitStatus(s)),
            Err(e) => Err(IoError::from_rtio_error(e)),
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
    /// # #![allow(experimental)]
    /// use std::io::process::{Command, ProcessExit};
    /// use std::io::IoResult;
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
    pub fn set_timeout(&mut self, timeout_ms: Option<u64>) {
        self.handle.set_timeout(timeout_ms)
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
        fn read(stream: Option<io::PipeStream>) -> Receiver<IoResult<Vec<u8>>> {
            let (tx, rx) = channel();
            match stream {
                Some(stream) => spawn(proc() {
                    let mut stream = stream;
                    tx.send(stream.read_to_end())
                }),
                None => tx.send(Ok(Vec::new()))
            }
            rx
        }
        let stdout = read(self.stdout.take());
        let stderr = read(self.stderr.take());

        let status = try!(self.wait());

        Ok(ProcessOutput {
            status: status,
            output: stdout.recv().ok().unwrap_or(Vec::new()),
            error:  stderr.recv().ok().unwrap_or(Vec::new()),
        })
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Close all I/O before exiting to ensure that the child doesn't wait
        // forever to print some text or something similar.
        drop(self.stdin.take());
        drop(self.stdout.take());
        drop(self.stderr.take());
        drop(mem::replace(&mut self.extra_io, Vec::new()));

        self.set_timeout(None);
        let _ = self.wait().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use io::process::{Command, Process};
    use prelude::*;

    // FIXME(#10380) these tests should not all be ignored on android.

    #[cfg(not(target_os="android"))]
    iotest!(fn smoke() {
        let p = Command::new("true").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().unwrap().success());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn smoke_failure() {
        match Command::new("if-this-is-a-binary-then-the-world-has-ended").spawn() {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn exit_reported_right() {
        let p = Command::new("false").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().unwrap().matches_exit_status(1));
        drop(p.wait().clone());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn signal_reported_right() {
        let p = Command::new("/bin/sh").arg("-c").arg("kill -1 $$").spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        match p.wait().unwrap() {
            process::ExitSignal(1) => {},
            result => fail!("not terminated by signal 1 (instead, {})", result),
        }
    })

    pub fn read_all(input: &mut Reader) -> String {
        input.read_to_str().unwrap()
    }

    pub fn run_output(cmd: Command) -> String {
        let p = cmd.spawn();
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.stdout.is_some());
        let ret = read_all(p.stdout.get_mut_ref() as &mut Reader);
        assert!(p.wait().unwrap().success());
        return ret;
    }

    #[cfg(not(target_os="android"))]
    iotest!(fn stdout_works() {
        let mut cmd = Command::new("echo");
        cmd.arg("foobar").stdout(CreatePipe(false, true));
        assert_eq!(run_output(cmd), "foobar\n".to_string());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn set_cwd_works() {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg("pwd")
           .cwd(&Path::new("/"))
           .stdout(CreatePipe(false, true));
        assert_eq!(run_output(cmd), "/\n".to_string());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn stdin_works() {
        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("read line; echo $line")
                            .stdin(CreatePipe(true, false))
                            .stdout(CreatePipe(false, true))
                            .spawn().unwrap();
        p.stdin.get_mut_ref().write("foobar".as_bytes()).unwrap();
        drop(p.stdin.take());
        let out = read_all(p.stdout.get_mut_ref() as &mut Reader);
        assert!(p.wait().unwrap().success());
        assert_eq!(out, "foobar\n".to_string());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn detach_works() {
        let mut p = Command::new("true").detached().spawn().unwrap();
        assert!(p.wait().unwrap().success());
    })

    #[cfg(windows)]
    iotest!(fn uid_fails_on_windows() {
        assert!(Command::new("test").uid(10).spawn().is_err());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn uid_works() {
        use libc;
        let mut p = Command::new("/bin/sh")
                            .arg("-c").arg("true")
                            .uid(unsafe { libc::getuid() as uint })
                            .gid(unsafe { libc::getgid() as uint })
                            .spawn().unwrap();
        assert!(p.wait().unwrap().success());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn uid_to_root_fails() {
        use libc;

        // if we're already root, this isn't a valid test. Most of the bots run
        // as non-root though (android is an exception).
        if unsafe { libc::getuid() == 0 } { return }
        assert!(Command::new("/bin/ls").uid(0).gid(0).spawn().is_err());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_process_status() {
        let mut status = Command::new("false").status().unwrap();
        assert!(status.matches_exit_status(1));

        status = Command::new("true").status().unwrap();
        assert!(status.success());
    })

    iotest!(fn test_process_output_fail_to_start() {
        match Command::new("/no-binary-by-this-name-should-exist").output() {
            Err(e) => assert_eq!(e.kind, FileNotFound),
            Ok(..) => fail!()
        }
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_process_output_output() {
        let ProcessOutput {status, output, error}
             = Command::new("echo").arg("hello").output().unwrap();
        let output_str = str::from_utf8(output.as_slice()).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello".to_string());
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_process_output_error() {
        let ProcessOutput {status, output, error}
             = Command::new("mkdir").arg(".").output().unwrap();

        assert!(status.matches_exit_status(1));
        assert_eq!(output, Vec::new());
        assert!(!error.is_empty());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_finish_once() {
        let mut prog = Command::new("false").spawn().unwrap();
        assert!(prog.wait().unwrap().matches_exit_status(1));
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_finish_twice() {
        let mut prog = Command::new("false").spawn().unwrap();
        assert!(prog.wait().unwrap().matches_exit_status(1));
        assert!(prog.wait().unwrap().matches_exit_status(1));
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_wait_with_output_once() {
        let prog = Command::new("echo").arg("hello").spawn().unwrap();
        let ProcessOutput {status, output, error} = prog.wait_with_output().unwrap();
        let output_str = str::from_utf8(output.as_slice()).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_string(), "hello".to_string());
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }
    })

    #[cfg(unix,not(target_os="android"))]
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

    iotest!(fn test_keep_current_working_dir() {
        use os;
        let prog = pwd_cmd().spawn().unwrap();

        let output = str::from_utf8(prog.wait_with_output().unwrap()
                                        .output.as_slice()).unwrap().to_string();
        let parent_dir = os::getcwd();
        let child_dir = Path::new(output.as_slice().trim());

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.unstable.device, child_stat.unstable.device);
        assert_eq!(parent_stat.unstable.inode, child_stat.unstable.inode);
    })

    iotest!(fn test_change_working_directory() {
        use os;
        // test changing to the parent of os::getcwd() because we know
        // the path exists (and os::getcwd() is not expected to be root)
        let parent_dir = os::getcwd().dir_path();
        let prog = pwd_cmd().cwd(&parent_dir).spawn().unwrap();

        let output = str::from_utf8(prog.wait_with_output().unwrap()
                                        .output.as_slice()).unwrap().to_string();
        let child_dir = Path::new(output.as_slice().trim().into_string());

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.unstable.device, child_stat.unstable.device);
        assert_eq!(parent_stat.unstable.inode, child_stat.unstable.inode);
    })

    #[cfg(unix,not(target_os="android"))]
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
    iotest!(fn test_inherit_env() {
        use os;
        if running_on_valgrind() { return; }

        let prog = env_cmd().spawn().unwrap();
        let output = str::from_utf8(prog.wait_with_output().unwrap()
                                        .output.as_slice()).unwrap().to_string();

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check windows magical empty-named variables
            assert!(k.is_empty() ||
                    output.as_slice()
                          .contains(format!("{}={}", *k, *v).as_slice()));
        }
    })
    #[cfg(target_os="android")]
    iotest!(fn test_inherit_env() {
        use os;
        if running_on_valgrind() { return; }

        let mut prog = env_cmd().spawn().unwrap();
        let output = str::from_utf8(prog.wait_with_output()
                                        .unwrap().output.as_slice())
                                   .unwrap().to_string();

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check android RANDOM variables
            if *k != "RANDOM".to_string() {
                assert!(output.as_slice()
                              .contains(format!("{}={}",
                                                *k,
                                                *v).as_slice()) ||
                        output.as_slice()
                              .contains(format!("{}=\'{}\'",
                                                *k,
                                                *v).as_slice()));
            }
        }
    })

    iotest!(fn test_add_to_env() {
        let new_env = box [("RUN_TEST_NEW_ENV", "123")];
        let prog = env_cmd().env(new_env).spawn().unwrap();
        let result = prog.wait_with_output().unwrap();
        let output = str::from_utf8_lossy(result.output.as_slice()).into_string();

        assert!(output.as_slice().contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    })

    #[cfg(unix)]
    pub fn sleeper() -> Process {
        Command::new("sleep").arg("1000").spawn().unwrap()
    }
    #[cfg(windows)]
    pub fn sleeper() -> Process {
        // There's a `timeout` command on windows, but it doesn't like having
        // its output piped, so instead just ping ourselves a few times with
        // gaps inbetweeen so we're sure this process is alive for awhile
        Command::new("ping").arg("127.0.0.1").arg("-n").arg("1000").spawn().unwrap()
    }

    iotest!(fn test_kill() {
        let mut p = sleeper();
        Process::kill(p.id(), PleaseExitSignal).unwrap();
        assert!(!p.wait().unwrap().success());
    })

    iotest!(fn test_exists() {
        let mut p = sleeper();
        assert!(Process::kill(p.id(), 0).is_ok());
        p.signal_kill().unwrap();
        assert!(!p.wait().unwrap().success());
    })

    iotest!(fn test_zero() {
        let mut p = sleeper();
        p.signal_kill().unwrap();
        for _ in range(0, 20) {
            if p.signal(0).is_err() {
                assert!(!p.wait().unwrap().success());
                return
            }
            timer::sleep(100);
        }
        fail!("never saw the child go away");
    })

    iotest!(fn wait_timeout() {
        let mut p = sleeper();
        p.set_timeout(Some(10));
        assert_eq!(p.wait().err().unwrap().kind, TimedOut);
        assert_eq!(p.wait().err().unwrap().kind, TimedOut);
        p.signal_kill().unwrap();
        p.set_timeout(None);
        assert!(p.wait().is_ok());
    })

    iotest!(fn wait_timeout2() {
        let (tx, rx) = channel();
        let tx2 = tx.clone();
        spawn(proc() {
            let mut p = sleeper();
            p.set_timeout(Some(10));
            assert_eq!(p.wait().err().unwrap().kind, TimedOut);
            p.signal_kill().unwrap();
            tx.send(());
        });
        spawn(proc() {
            let mut p = sleeper();
            p.set_timeout(Some(10));
            assert_eq!(p.wait().err().unwrap().kind, TimedOut);
            p.signal_kill().unwrap();
            tx2.send(());
        });
        rx.recv();
        rx.recv();
    })
}
