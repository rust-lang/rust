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

use prelude::*;

use fmt;
use io::IoResult;
use io;
use libc;
use mem;
use rt::rtio::{RtioProcess, IoFactory, LocalIo};

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
/// This structure is used to create, run, and manage child processes. A process
/// is configured with the `ProcessConfig` struct which contains specific
/// options for dictating how the child is spawned.
///
/// # Example
///
/// ```should_fail
/// use std::io::Process;
///
/// let mut child = match Process::new("/bin/cat", ["file.txt".to_owned()]) {
///     Ok(child) => child,
///     Err(e) => fail!("failed to execute child: {}", e),
/// };
///
/// let contents = child.stdout.get_mut_ref().read_to_end();
/// assert!(child.wait().success());
/// ```
pub struct Process {
    handle: ~RtioProcess:Send,

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
    pub extra_io: ~[Option<io::PipeStream>],
}

/// This configuration describes how a new process should be spawned. A blank
/// configuration can be created with `ProcessConfig::new()`. It is also
/// recommented to use a functional struct update pattern when creating process
/// configuration:
///
/// ```
/// use std::io::ProcessConfig;
///
/// let config = ProcessConfig {
///     program: "/bin/sh",
///     args: &["-c".to_owned(), "true".to_owned()],
///     .. ProcessConfig::new()
/// };
/// ```
pub struct ProcessConfig<'a> {
    /// Path to the program to run
    pub program: &'a str,

    /// Arguments to pass to the program (doesn't include the program itself)
    pub args: &'a [~str],

    /// Optional environment to specify for the program. If this is None, then
    /// it will inherit the current process's environment.
    pub env: Option<&'a [(~str, ~str)]>,

    /// Optional working directory for the new process. If this is None, then
    /// the current directory of the running process is inherited.
    pub cwd: Option<&'a Path>,

    /// Configuration for the child process's stdin handle (file descriptor 0).
    /// This field defaults to `CreatePipe(true, false)` so the input can be
    /// written to.
    pub stdin: StdioContainer,

    /// Configuration for the child process's stdout handle (file descriptor 1).
    /// This field defaults to `CreatePipe(false, true)` so the output can be
    /// collected.
    pub stdout: StdioContainer,

    /// Configuration for the child process's stdout handle (file descriptor 2).
    /// This field defaults to `CreatePipe(false, true)` so the output can be
    /// collected.
    pub stderr: StdioContainer,

    /// Any number of streams/file descriptors/pipes may be attached to this
    /// process. This list enumerates the file descriptors and such for the
    /// process to be spawned, and the file descriptors inherited will start at
    /// 3 and go to the length of this array. The first three file descriptors
    /// (stdin/stdout/stderr) are configured with the `stdin`, `stdout`, and
    /// `stderr` fields.
    pub extra_io: &'a [StdioContainer],

    /// Sets the child process's user id. This translates to a `setuid` call in
    /// the child process. Setting this value on windows will cause the spawn to
    /// fail. Failure in the `setuid` call on unix will also cause the spawn to
    /// fail.
    pub uid: Option<uint>,

    /// Similar to `uid`, but sets the group id of the child process. This has
    /// the same semantics as the `uid` field.
    pub gid: Option<uint>,

    /// If true, the child process is spawned in a detached state. On unix, this
    /// means that the child is the leader of a new process group.
    pub detach: bool,
}

/// The output of a finished process.
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
#[deriving(Eq, TotalEq, Clone)]
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
            ExitStatus(code) =>  write!(f.buf, "exit code: {}", code),
            ExitSignal(code) =>  write!(f.buf, "signal: {}", code),
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

impl<'a> ProcessConfig<'a> {
    /// Creates a new configuration with blanks as all of the defaults. This is
    /// useful when using functional struct updates:
    ///
    /// ```rust
    /// use std::io::process::{ProcessConfig, Process};
    ///
    /// let config = ProcessConfig {
    ///     program: "/bin/sh",
    ///     args: &["-c".to_owned(), "echo hello".to_owned()],
    ///     .. ProcessConfig::new()
    /// };
    ///
    /// let p = Process::configure(config);
    /// ```
    ///
    pub fn new<'a>() -> ProcessConfig<'a> {
        ProcessConfig {
            program: "",
            args: &[],
            env: None,
            cwd: None,
            stdin: CreatePipe(true, false),
            stdout: CreatePipe(false, true),
            stderr: CreatePipe(false, true),
            extra_io: &[],
            uid: None,
            gid: None,
            detach: false,
        }
    }
}

impl Process {
    /// Creates a new process for the specified program/arguments, using
    /// otherwise default configuration.
    ///
    /// By default, new processes have their stdin/stdout/stderr handles created
    /// as pipes the can be manipulated through the respective fields of the
    /// returned `Process`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::io::Process;
    ///
    /// let mut process = match Process::new("sh", &["c".to_owned(), "echo hello".to_owned()]) {
    ///     Ok(p) => p,
    ///     Err(e) => fail!("failed to execute process: {}", e),
    /// };
    ///
    /// let output = process.stdout.get_mut_ref().read_to_end();
    /// ```
    pub fn new(prog: &str, args: &[~str]) -> IoResult<Process> {
        Process::configure(ProcessConfig {
            program: prog,
            args: args,
            .. ProcessConfig::new()
        })
    }

    /// Executes the specified program with arguments, waiting for it to finish
    /// and collecting all of its output.
    ///
    /// # Example
    ///
    /// ```
    /// use std::io::Process;
    /// use std::str;
    ///
    /// let output = match Process::output("cat", ["foo.txt".to_owned()]) {
    ///     Ok(output) => output,
    ///     Err(e) => fail!("failed to execute process: {}", e),
    /// };
    ///
    /// println!("status: {}", output.status);
    /// println!("stdout: {}", str::from_utf8_lossy(output.output.as_slice()));
    /// println!("stderr: {}", str::from_utf8_lossy(output.error.as_slice()));
    /// ```
    pub fn output(prog: &str, args: &[~str]) -> IoResult<ProcessOutput> {
        Process::new(prog, args).map(|mut p| p.wait_with_output())
    }

    /// Executes a child process and collects its exit status. This will block
    /// waiting for the child to exit.
    ///
    /// # Example
    ///
    /// ```
    /// use std::io::Process;
    ///
    /// let status = match Process::status("ls", []) {
    ///     Ok(status) => status,
    ///     Err(e) => fail!("failed to execute process: {}", e),
    /// };
    ///
    /// println!("process exited with: {}", status);
    /// ```
    pub fn status(prog: &str, args: &[~str]) -> IoResult<ProcessExit> {
        Process::new(prog, args).map(|mut p| p.wait())
    }

    /// Creates a new process with the specified configuration.
    pub fn configure(config: ProcessConfig) -> IoResult<Process> {
        let mut config = Some(config);
        LocalIo::maybe_raise(|io| {
            io.spawn(config.take_unwrap()).map(|(p, io)| {
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
        })
    }

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
        LocalIo::maybe_raise(|io| io.kill(id, signal))
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
        self.handle.kill(signal)
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
    pub fn wait(&mut self) -> ProcessExit {
        drop(self.stdin.take());
        self.handle.wait()
    }

    /// Simultaneously wait for the child to exit and collect all remaining
    /// output on the stdout/stderr handles, returning a `ProcessOutput`
    /// instance.
    ///
    /// The stdin handle to the child is closed before waiting.
    pub fn wait_with_output(&mut self) -> ProcessOutput {
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

        let status = self.wait();

        ProcessOutput { status: status,
                        output: stdout.recv().ok().unwrap_or(Vec::new()),
                        error:  stderr.recv().ok().unwrap_or(Vec::new()) }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Close all I/O before exiting to ensure that the child doesn't wait
        // forever to print some text or something similar.
        drop(self.stdin.take());
        drop(self.stdout.take());
        drop(self.stderr.take());
        drop(mem::replace(&mut self.extra_io, ~[]));

        self.wait();
    }
}

#[cfg(test)]
mod tests {
    use io::process::{ProcessConfig, Process};
    use prelude::*;
    use str::StrSlice;

    // FIXME(#10380) these tests should not all be ignored on android.

    #[cfg(not(target_os="android"))]
    iotest!(fn smoke() {
        let args = ProcessConfig {
            program: "true",
            .. ProcessConfig::new()
        };
        let p = Process::configure(args);
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().success());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn smoke_failure() {
        let args = ProcessConfig {
            program: "if-this-is-a-binary-then-the-world-has-ended",
            .. ProcessConfig::new()
        };
        match Process::configure(args) {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn exit_reported_right() {
        let args = ProcessConfig {
            program: "false",
            .. ProcessConfig::new()
        };
        let p = Process::configure(args);
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.wait().matches_exit_status(1));
        drop(p.wait().clone());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn signal_reported_right() {
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &["-c".to_owned(), "kill -1 $$".to_owned()],
            .. ProcessConfig::new()
        };
        let p = Process::configure(args);
        assert!(p.is_ok());
        let mut p = p.unwrap();
        match p.wait() {
            process::ExitSignal(1) => {},
            result => fail!("not terminated by signal 1 (instead, {})", result),
        }
    })

    pub fn read_all(input: &mut Reader) -> ~str {
        input.read_to_str().unwrap()
    }

    pub fn run_output(args: ProcessConfig) -> ~str {
        let p = Process::configure(args);
        assert!(p.is_ok());
        let mut p = p.unwrap();
        assert!(p.stdout.is_some());
        let ret = read_all(p.stdout.get_mut_ref() as &mut Reader);
        assert!(p.wait().success());
        return ret;
    }

    #[cfg(not(target_os="android"))]
    iotest!(fn stdout_works() {
        let args = ProcessConfig {
            program: "echo",
            args: &["foobar".to_owned()],
            stdout: CreatePipe(false, true),
            .. ProcessConfig::new()
        };
        assert_eq!(run_output(args), "foobar\n".to_owned());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn set_cwd_works() {
        let cwd = Path::new("/");
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &["-c".to_owned(), "pwd".to_owned()],
            cwd: Some(&cwd),
            stdout: CreatePipe(false, true),
            .. ProcessConfig::new()
        };
        assert_eq!(run_output(args), "/\n".to_owned());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn stdin_works() {
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &["-c".to_owned(), "read line; echo $line".to_owned()],
            stdin: CreatePipe(true, false),
            stdout: CreatePipe(false, true),
            .. ProcessConfig::new()
        };
        let mut p = Process::configure(args).unwrap();
        p.stdin.get_mut_ref().write("foobar".as_bytes()).unwrap();
        drop(p.stdin.take());
        let out = read_all(p.stdout.get_mut_ref() as &mut Reader);
        assert!(p.wait().success());
        assert_eq!(out, "foobar\n".to_owned());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn detach_works() {
        let args = ProcessConfig {
            program: "true",
            detach: true,
            .. ProcessConfig::new()
        };
        let mut p = Process::configure(args).unwrap();
        assert!(p.wait().success());
    })

    #[cfg(windows)]
    iotest!(fn uid_fails_on_windows() {
        let args = ProcessConfig {
            program: "test",
            uid: Some(10),
            .. ProcessConfig::new()
        };
        assert!(Process::configure(args).is_err());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn uid_works() {
        use libc;
        let args = ProcessConfig {
            program: "/bin/sh",
            args: &["-c".to_owned(), "true".to_owned()],
            uid: Some(unsafe { libc::getuid() as uint }),
            gid: Some(unsafe { libc::getgid() as uint }),
            .. ProcessConfig::new()
        };
        let mut p = Process::configure(args).unwrap();
        assert!(p.wait().success());
    })

    #[cfg(unix, not(target_os="android"))]
    iotest!(fn uid_to_root_fails() {
        use libc;

        // if we're already root, this isn't a valid test. Most of the bots run
        // as non-root though (android is an exception).
        if unsafe { libc::getuid() == 0 } { return }
        let args = ProcessConfig {
            program: "/bin/ls",
            uid: Some(0),
            gid: Some(0),
            .. ProcessConfig::new()
        };
        assert!(Process::configure(args).is_err());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_process_status() {
        let mut status = Process::status("false", []).unwrap();
        assert!(status.matches_exit_status(1));

        status = Process::status("true", []).unwrap();
        assert!(status.success());
    })

    iotest!(fn test_process_output_fail_to_start() {
        match Process::output("/no-binary-by-this-name-should-exist", []) {
            Err(e) => assert_eq!(e.kind, FileNotFound),
            Ok(..) => fail!()
        }
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_process_output_output() {

        let ProcessOutput {status, output, error}
             = Process::output("echo", ["hello".to_owned()]).unwrap();
        let output_str = str::from_utf8(output.as_slice()).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_owned(), "hello".to_owned());
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_process_output_error() {
        let ProcessOutput {status, output, error}
             = Process::output("mkdir", [".".to_owned()]).unwrap();

        assert!(status.matches_exit_status(1));
        assert_eq!(output, Vec::new());
        assert!(!error.is_empty());
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_finish_once() {
        let mut prog = Process::new("false", []).unwrap();
        assert!(prog.wait().matches_exit_status(1));
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_finish_twice() {
        let mut prog = Process::new("false", []).unwrap();
        assert!(prog.wait().matches_exit_status(1));
        assert!(prog.wait().matches_exit_status(1));
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_wait_with_output_once() {

        let mut prog = Process::new("echo", ["hello".to_owned()]).unwrap();
        let ProcessOutput {status, output, error} = prog.wait_with_output();
        let output_str = str::from_utf8(output.as_slice()).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_owned(), "hello".to_owned());
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }
    })

    #[cfg(not(target_os="android"))]
    iotest!(fn test_wait_with_output_twice() {
        let mut prog = Process::new("echo", ["hello".to_owned()]).unwrap();
        let ProcessOutput {status, output, error} = prog.wait_with_output();

        let output_str = str::from_utf8(output.as_slice()).unwrap();

        assert!(status.success());
        assert_eq!(output_str.trim().to_owned(), "hello".to_owned());
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }

        let ProcessOutput {status, output, error} = prog.wait_with_output();

        assert!(status.success());
        assert_eq!(output, Vec::new());
        // FIXME #7224
        if !running_on_valgrind() {
            assert_eq!(error, Vec::new());
        }
    })

    #[cfg(unix,not(target_os="android"))]
    pub fn run_pwd(dir: Option<&Path>) -> Process {
        Process::configure(ProcessConfig {
            program: "pwd",
            cwd: dir,
            .. ProcessConfig::new()
        }).unwrap()
    }
    #[cfg(target_os="android")]
    pub fn run_pwd(dir: Option<&Path>) -> Process {
        Process::configure(ProcessConfig {
            program: "/system/bin/sh",
            args: &["-c".to_owned(),"pwd".to_owned()],
            cwd: dir.map(|a| &*a),
            .. ProcessConfig::new()
        }).unwrap()
    }

    #[cfg(windows)]
    pub fn run_pwd(dir: Option<&Path>) -> Process {
        Process::configure(ProcessConfig {
            program: "cmd",
            args: &["/c".to_owned(), "cd".to_owned()],
            cwd: dir.map(|a| &*a),
            .. ProcessConfig::new()
        }).unwrap()
    }

    iotest!(fn test_keep_current_working_dir() {
        use os;
        let mut prog = run_pwd(None);

        let output = str::from_utf8(prog.wait_with_output().output.as_slice()).unwrap().to_owned();
        let parent_dir = os::getcwd();
        let child_dir = Path::new(output.trim());

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
        let mut prog = run_pwd(Some(&parent_dir));

        let output = str::from_utf8(prog.wait_with_output().output.as_slice()).unwrap().to_owned();
        let child_dir = Path::new(output.trim());

        let parent_stat = parent_dir.stat().unwrap();
        let child_stat = child_dir.stat().unwrap();

        assert_eq!(parent_stat.unstable.device, child_stat.unstable.device);
        assert_eq!(parent_stat.unstable.inode, child_stat.unstable.inode);
    })

    #[cfg(unix,not(target_os="android"))]
    pub fn run_env(env: Option<~[(~str, ~str)]>) -> Process {
        Process::configure(ProcessConfig {
            program: "env",
            env: env.as_ref().map(|e| e.as_slice()),
            .. ProcessConfig::new()
        }).unwrap()
    }
    #[cfg(target_os="android")]
    pub fn run_env(env: Option<~[(~str, ~str)]>) -> Process {
        Process::configure(ProcessConfig {
            program: "/system/bin/sh",
            args: &["-c".to_owned(),"set".to_owned()],
            env: env.as_ref().map(|e| e.as_slice()),
            .. ProcessConfig::new()
        }).unwrap()
    }

    #[cfg(windows)]
    pub fn run_env(env: Option<~[(~str, ~str)]>) -> Process {
        Process::configure(ProcessConfig {
            program: "cmd",
            args: &["/c".to_owned(), "set".to_owned()],
            env: env.as_ref().map(|e| e.as_slice()),
            .. ProcessConfig::new()
        }).unwrap()
    }

    #[cfg(not(target_os="android"))]
    iotest!(fn test_inherit_env() {
        use os;
        if running_on_valgrind() { return; }

        let mut prog = run_env(None);
        let output = str::from_utf8(prog.wait_with_output().output.as_slice()).unwrap().to_owned();

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check windows magical empty-named variables
            assert!(k.is_empty() || output.contains(format!("{}={}", *k, *v)));
        }
    })
    #[cfg(target_os="android")]
    iotest!(fn test_inherit_env() {
        use os;
        if running_on_valgrind() { return; }

        let mut prog = run_env(None);
        let output = str::from_utf8(prog.wait_with_output().output.as_slice()).unwrap().to_owned();

        let r = os::env();
        for &(ref k, ref v) in r.iter() {
            // don't check android RANDOM variables
            if *k != "RANDOM".to_owned() {
                assert!(output.contains(format!("{}={}", *k, *v)) ||
                        output.contains(format!("{}=\'{}\'", *k, *v)));
            }
        }
    })

    iotest!(fn test_add_to_env() {
        let new_env = ~[("RUN_TEST_NEW_ENV".to_owned(), "123".to_owned())];

        let mut prog = run_env(Some(new_env));
        let result = prog.wait_with_output();
        let output = str::from_utf8_lossy(result.output.as_slice()).into_owned();

        assert!(output.contains("RUN_TEST_NEW_ENV=123"),
                "didn't find RUN_TEST_NEW_ENV inside of:\n\n{}", output);
    })

    #[cfg(unix)]
    pub fn sleeper() -> Process {
        Process::new("sleep", ["1000".to_owned()]).unwrap()
    }
    #[cfg(windows)]
    pub fn sleeper() -> Process {
        // There's a `timeout` command on windows, but it doesn't like having
        // its output piped, so instead just ping ourselves a few times with
        // gaps inbetweeen so we're sure this process is alive for awhile
        Process::new("ping", ["127.0.0.1".to_owned(), "-n".to_owned(), "1000".to_owned()]).unwrap()
    }

    iotest!(fn test_kill() {
        let mut p = sleeper();
        Process::kill(p.id(), PleaseExitSignal).unwrap();
        assert!(!p.wait().success());
    })

    iotest!(fn test_exists() {
        let mut p = sleeper();
        assert!(Process::kill(p.id(), 0).is_ok());
        p.signal_kill().unwrap();
        assert!(!p.wait().success());
    })

    iotest!(fn test_zero() {
        let mut p = sleeper();
        p.signal_kill().unwrap();
        for _ in range(0, 20) {
            if p.signal(0).is_err() {
                assert!(!p.wait().success());
                return
            }
            timer::sleep(100);
        }
        fail!("never saw the child go away");
    })
}
