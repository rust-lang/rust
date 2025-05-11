use std::ffi::OsStr;
use std::io::Write;
use std::path::Path;
use std::process::{Command as StdCommand, ExitStatus, Output, Stdio};
use std::{ffi, panic};

use build_helper::drop_bomb::DropBomb;

use crate::util::handle_failed_output;
use crate::{
    assert_contains, assert_contains_regex, assert_equals, assert_not_contains,
    assert_not_contains_regex,
};

/// This is a custom command wrapper that simplifies working with commands and makes it easier to
/// ensure that we check the exit status of executed processes.
///
/// # A [`Command`] must be executed exactly once
///
/// A [`Command`] is armed by a [`DropBomb`] on construction to enforce that it will be executed. If
/// a [`Command`] is constructed but never executed, the drop bomb will explode and cause the test
/// to panic. Execution methods [`run`] and [`run_fail`] will defuse the drop bomb. A test
/// containing constructed but never executed commands is dangerous because it can give a false
/// sense of confidence.
///
/// Each [`Command`] invocation can also only be executed once, because we want to enforce
/// `std{in,out,err}` config via [`std::process::Stdio`] but [`std::process::Stdio`] is not
/// cloneable.
///
/// In this sense, [`Command`] exhibits linear type semantics but enforced at run-time.
///
/// [`run`]: Self::run
/// [`run_fail`]: Self::run_fail
/// [`run_unchecked`]: Self::run_unchecked
#[derive(Debug)]
pub struct Command {
    cmd: StdCommand,
    // Convience for providing a quick stdin buffer.
    stdin_buf: Option<Box<[u8]>>,

    // Configurations for child process's std{in,out,err} handles.
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,

    // Emulate linear type semantics.
    drop_bomb: DropBomb,
    already_executed: bool,
}

impl Command {
    #[track_caller]
    pub fn new<P: AsRef<OsStr>>(program: P) -> Self {
        let program = program.as_ref();
        Self {
            cmd: StdCommand::new(program),
            stdin_buf: None,
            drop_bomb: DropBomb::arm(program),
            stdin: None,
            stdout: None,
            stderr: None,
            already_executed: false,
        }
    }

    // Internal-only.
    pub(crate) fn into_raw_command(mut self) -> std::process::Command {
        self.drop_bomb.defuse();
        self.cmd
    }

    /// Specify a stdin input buffer. This is a convenience helper,
    pub fn stdin_buf<I: AsRef<[u8]>>(&mut self, input: I) -> &mut Self {
        self.stdin_buf = Some(input.as_ref().to_vec().into_boxed_slice());
        self
    }

    /// Configuration for the child process’s standard input (stdin) handle.
    ///
    /// See [`std::process::Command::stdin`].
    pub fn stdin<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Self {
        self.stdin = Some(cfg.into());
        self
    }

    /// Configuration for the child process’s standard output (stdout) handle.
    ///
    /// See [`std::process::Command::stdout`].
    pub fn stdout<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Self {
        self.stdout = Some(cfg.into());
        self
    }

    /// Configuration for the child process’s standard error (stderr) handle.
    ///
    /// See [`std::process::Command::stderr`].
    pub fn stderr<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Self {
        self.stderr = Some(cfg.into());
        self
    }

    /// Specify an environment variable.
    pub fn env<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: AsRef<ffi::OsStr>,
        V: AsRef<ffi::OsStr>,
    {
        self.cmd.env(key, value);
        self
    }

    /// Remove an environmental variable.
    pub fn env_remove<K>(&mut self, key: K) -> &mut Self
    where
        K: AsRef<ffi::OsStr>,
    {
        self.cmd.env_remove(key);
        self
    }

    /// Generic command argument provider. Prefer specific helper methods if possible.
    /// Note that for some executables, arguments might be platform specific. For C/C++
    /// compilers, arguments might be platform *and* compiler specific.
    pub fn arg<S>(&mut self, arg: S) -> &mut Self
    where
        S: AsRef<ffi::OsStr>,
    {
        self.cmd.arg(arg);
        self
    }

    /// Generic command arguments provider. Prefer specific helper methods if possible.
    /// Note that for some executables, arguments might be platform specific. For C/C++
    /// compilers, arguments might be platform *and* compiler specific.
    pub fn args<S, V>(&mut self, args: V) -> &mut Self
    where
        S: AsRef<ffi::OsStr>,
        V: AsRef<[S]>,
    {
        self.cmd.args(args.as_ref());
        self
    }

    /// Inspect what the underlying [`std::process::Command`] is up to the
    /// current construction.
    pub fn inspect<I>(&mut self, inspector: I) -> &mut Self
    where
        I: FnOnce(&StdCommand),
    {
        inspector(&self.cmd);
        self
    }

    /// Set the path where the command will be run.
    pub fn current_dir<P: AsRef<Path>>(&mut self, path: P) -> &mut Self {
        self.cmd.current_dir(path);
        self
    }

    /// Set an auxiliary stream passed to the process, besides the stdio streams.
    ///
    /// # Notes
    ///
    /// Use with caution! Ideally, only set one aux fd; if there are multiple, their old `fd` may
    /// overlap with another's `new_fd`, and may break. The caller must make sure this is not the
    /// case. This function is only "safe" because the safety requirements are practically not
    /// possible to uphold.
    #[cfg(unix)]
    pub fn set_aux_fd<F: Into<std::os::fd::OwnedFd>>(
        &mut self,
        new_fd: std::os::fd::RawFd,
        fd: F,
    ) -> &mut Self {
        use std::mem;
        // NOTE: If more than 1 auxiliary file descriptor is needed, this function should be
        // rewritten.
        use std::os::fd::AsRawFd;
        use std::os::unix::process::CommandExt;

        let cvt = |x| if x == -1 { Err(std::io::Error::last_os_error()) } else { Ok(()) };

        // Ensure fd stays open until the fork.
        let fd = mem::ManuallyDrop::new(fd.into());
        let fd = fd.as_raw_fd();

        if fd == new_fd {
            // If the new file descriptor is already the same as fd, just turn off `FD_CLOEXEC`.
            let fd_flags = {
                let ret = unsafe { libc::fcntl(fd, libc::F_GETFD, 0) };
                if ret < 0 {
                    panic!("failed to read fd flags: {}", std::io::Error::last_os_error());
                }
                ret
            };
            // Clear `FD_CLOEXEC`.
            let fd_flags = fd_flags & !libc::FD_CLOEXEC;

            // SAFETY(io-safety): `fd` is already owned.
            cvt(unsafe { libc::fcntl(fd, libc::F_SETFD, fd_flags as libc::c_int) })
                .expect("disabling CLOEXEC failed");
        }
        let pre_exec = move || {
            if fd.as_raw_fd() != new_fd {
                // SAFETY(io-safety): it's the caller's responsibility that we won't override the
                // target fd.
                cvt(unsafe { libc::dup2(fd, new_fd) })?;
            }
            Ok(())
        };
        // SAFETY(pre-exec-safe): `dup2` is pre-exec-safe.
        unsafe { self.cmd.pre_exec(pre_exec) };
        self
    }

    /// Run the constructed command and assert that it is successfully run.
    ///
    /// By default, std{in,out,err} are [`Stdio::piped()`].
    #[track_caller]
    pub fn run(&mut self) -> CompletedProcess {
        let output = self.command_output();
        if !output.status().success() {
            handle_failed_output(&self, output, panic::Location::caller().line());
        }
        output
    }

    /// Run the constructed command and assert that it does not successfully run.
    ///
    /// By default, std{in,out,err} are [`Stdio::piped()`].
    #[track_caller]
    pub fn run_fail(&mut self) -> CompletedProcess {
        let output = self.command_output();
        if output.status().success() {
            handle_failed_output(&self, output, panic::Location::caller().line());
        }
        output
    }

    /// Run the command but do not check its exit status. Only use if you explicitly don't care
    /// about the exit status.
    ///
    /// Prefer to use [`Self::run`] and [`Self::run_fail`] whenever possible.
    #[track_caller]
    pub fn run_unchecked(&mut self) -> CompletedProcess {
        self.command_output()
    }

    #[track_caller]
    fn command_output(&mut self) -> CompletedProcess {
        if self.already_executed {
            panic!("command was already executed");
        } else {
            self.already_executed = true;
        }

        self.drop_bomb.defuse();
        // let's make sure we piped all the input and outputs
        self.cmd.stdin(self.stdin.take().unwrap_or(Stdio::piped()));
        self.cmd.stdout(self.stdout.take().unwrap_or(Stdio::piped()));
        self.cmd.stderr(self.stderr.take().unwrap_or(Stdio::piped()));

        let output = if let Some(input) = &self.stdin_buf {
            let mut child = self.cmd.spawn().unwrap();

            {
                let mut stdin = child.stdin.take().unwrap();
                stdin.write_all(input.as_ref()).unwrap();
            }

            child.wait_with_output().expect("failed to get output of finished process")
        } else {
            self.cmd.output().expect("failed to get output of finished process")
        };
        output.into()
    }
}

/// Represents the result of an executed process.
/// The various `assert_` helper methods should preferably be used for
/// checking the contents of stdout/stderr.
pub struct CompletedProcess {
    output: Output,
}

impl CompletedProcess {
    #[must_use]
    #[track_caller]
    pub fn stdout(&self) -> Vec<u8> {
        self.output.stdout.clone()
    }

    #[must_use]
    #[track_caller]
    pub fn stdout_utf8(&self) -> String {
        String::from_utf8(self.output.stdout.clone()).expect("stdout is not valid UTF-8")
    }

    #[must_use]
    #[track_caller]
    pub fn invalid_stdout_utf8(&self) -> String {
        String::from_utf8_lossy(&self.output.stdout.clone()).to_string()
    }

    #[must_use]
    #[track_caller]
    pub fn stderr(&self) -> Vec<u8> {
        self.output.stderr.clone()
    }

    #[must_use]
    #[track_caller]
    pub fn stderr_utf8(&self) -> String {
        String::from_utf8(self.output.stderr.clone()).expect("stderr is not valid UTF-8")
    }

    #[must_use]
    #[track_caller]
    pub fn invalid_stderr_utf8(&self) -> String {
        String::from_utf8_lossy(&self.output.stderr.clone()).to_string()
    }

    #[must_use]
    pub fn status(&self) -> ExitStatus {
        self.output.status
    }

    /// Checks that trimmed `stdout` matches trimmed `expected`.
    #[track_caller]
    pub fn assert_stdout_equals<S: AsRef<str>>(&self, expected: S) -> &Self {
        assert_equals(self.stdout_utf8().trim(), expected.as_ref().trim());
        self
    }

    /// Checks that `stdout` does not contain `unexpected`.
    #[track_caller]
    pub fn assert_stdout_not_contains<S: AsRef<str>>(&self, unexpected: S) -> &Self {
        assert_not_contains(&self.stdout_utf8(), unexpected);
        self
    }

    /// Checks that `stdout` does not contain the regex pattern `unexpected`.
    #[track_caller]
    pub fn assert_stdout_not_contains_regex<S: AsRef<str>>(&self, unexpected: S) -> &Self {
        assert_not_contains_regex(&self.stdout_utf8(), unexpected);
        self
    }

    /// Checks that `stdout` contains `expected`.
    #[track_caller]
    pub fn assert_stdout_contains<S: AsRef<str>>(&self, expected: S) -> &Self {
        assert_contains(&self.stdout_utf8(), expected);
        self
    }

    /// Checks that `stdout` contains the regex pattern `expected`.
    #[track_caller]
    pub fn assert_stdout_contains_regex<S: AsRef<str>>(&self, expected: S) -> &Self {
        assert_contains_regex(&self.stdout_utf8(), expected);
        self
    }

    /// Checks that trimmed `stderr` matches trimmed `expected`.
    #[track_caller]
    pub fn assert_stderr_equals<S: AsRef<str>>(&self, expected: S) -> &Self {
        assert_equals(self.stderr_utf8().trim(), expected.as_ref().trim());
        self
    }

    /// Checks that `stderr` contains `expected`.
    #[track_caller]
    pub fn assert_stderr_contains<S: AsRef<str>>(&self, expected: S) -> &Self {
        assert_contains(&self.stderr_utf8(), expected);
        self
    }

    /// Checks that `stderr` contains the regex pattern `expected`.
    #[track_caller]
    pub fn assert_stderr_contains_regex<S: AsRef<str>>(&self, expected: S) -> &Self {
        assert_contains_regex(&self.stderr_utf8(), expected);
        self
    }

    /// Checks that `stderr` does not contain `unexpected`.
    #[track_caller]
    pub fn assert_stderr_not_contains<S: AsRef<str>>(&self, unexpected: S) -> &Self {
        assert_not_contains(&self.stderr_utf8(), unexpected);
        self
    }

    /// Checks that `stderr` does not contain the regex pattern `unexpected`.
    #[track_caller]
    pub fn assert_stderr_not_contains_regex<S: AsRef<str>>(&self, unexpected: S) -> &Self {
        assert_not_contains_regex(&self.stderr_utf8(), unexpected);
        self
    }

    /// Check the **exit status** of the process. On Unix, this is *not* the **wait status**.
    ///
    /// See [`std::process::ExitStatus::code`]. This is not to be confused with
    /// [`std::process::ExitCode`].
    #[track_caller]
    pub fn assert_exit_code(&self, code: i32) -> &Self {
        assert_eq!(self.output.status.code(), Some(code));
        self
    }
}

impl From<Output> for CompletedProcess {
    fn from(output: Output) -> Self {
        Self { output }
    }
}
