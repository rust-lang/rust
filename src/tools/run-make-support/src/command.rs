use std::ffi;
use std::ffi::OsStr;
use std::io::Write;
use std::panic;
use std::path::Path;
use std::process::{Command as StdCommand, ExitStatus, Output, Stdio};

use crate::drop_bomb::DropBomb;
use crate::{assert_contains, assert_equals, assert_not_contains, handle_failed_output};

/// This is a custom command wrapper that simplifies working with commands and makes it easier to
/// ensure that we check the exit status of executed processes.
///
/// # A [`Command`] must be executed
///
/// A [`Command`] is armed by a [`DropBomb`] on construction to enforce that it will be executed. If
/// a [`Command`] is constructed but never executed, the drop bomb will explode and cause the test
/// to panic. Execution methods [`run`] and [`run_fail`] will defuse the drop bomb. A test
/// containing constructed but never executed commands is dangerous because it can give a false
/// sense of confidence.
///
/// [`run`]: Self::run
/// [`run_fail`]: Self::run_fail
/// [`run_unchecked`]: Self::run_unchecked
#[derive(Debug)]
pub struct Command {
    cmd: StdCommand,
    stdin: Option<Box<[u8]>>,
    drop_bomb: DropBomb,
}

impl Command {
    #[track_caller]
    pub fn new<P: AsRef<OsStr>>(program: P) -> Self {
        let program = program.as_ref();
        Self { cmd: StdCommand::new(program), stdin: None, drop_bomb: DropBomb::arm(program) }
    }

    pub fn set_stdin(&mut self, stdin: Box<[u8]>) {
        self.stdin = Some(stdin);
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
    pub fn args<S>(&mut self, args: &[S]) -> &mut Self
    where
        S: AsRef<ffi::OsStr>,
    {
        self.cmd.args(args);
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

    /// Run the constructed command and assert that it is successfully run.
    #[track_caller]
    pub fn run(&mut self) -> CompletedProcess {
        let output = self.command_output();
        if !output.status().success() {
            handle_failed_output(&self, output, panic::Location::caller().line());
        }
        output
    }

    /// Run the constructed command and assert that it does not successfully run.
    #[track_caller]
    pub fn run_fail(&mut self) -> CompletedProcess {
        let output = self.command_output();
        if output.status().success() {
            handle_failed_output(&self, output, panic::Location::caller().line());
        }
        output
    }

    /// Run the command but do not check its exit status.
    /// Only use if you explicitly don't care about the exit status.
    /// Prefer to use [`Self::run`] and [`Self::run_fail`]
    /// whenever possible.
    #[track_caller]
    pub fn run_unchecked(&mut self) -> CompletedProcess {
        self.command_output()
    }

    #[track_caller]
    fn command_output(&mut self) -> CompletedProcess {
        self.drop_bomb.defuse();
        // let's make sure we piped all the input and outputs
        self.cmd.stdin(Stdio::piped());
        self.cmd.stdout(Stdio::piped());
        self.cmd.stderr(Stdio::piped());

        let output = if let Some(input) = &self.stdin {
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
    pub fn stdout_utf8(&self) -> String {
        String::from_utf8(self.output.stdout.clone()).expect("stdout is not valid UTF-8")
    }

    #[must_use]
    pub fn stderr_utf8(&self) -> String {
        String::from_utf8(self.output.stderr.clone()).expect("stderr is not valid UTF-8")
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
        assert_not_contains(&self.stdout_utf8(), unexpected.as_ref());
        self
    }

    /// Checks that `stdout` contains `expected`.
    #[track_caller]
    pub fn assert_stdout_contains<S: AsRef<str>>(&self, expected: S) -> &Self {
        assert_contains(&self.stdout_utf8(), expected.as_ref());
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
        assert_contains(&self.stderr_utf8(), expected.as_ref());
        self
    }

    /// Checks that `stderr` does not contain `unexpected`.
    #[track_caller]
    pub fn assert_stderr_not_contains<S: AsRef<str>>(&self, unexpected: S) -> &Self {
        assert_not_contains(&self.stdout_utf8(), unexpected.as_ref());
        self
    }

    #[track_caller]
    pub fn assert_exit_code(&self, code: i32) -> &Self {
        assert!(self.output.status.code() == Some(code));
        self
    }
}

impl From<Output> for CompletedProcess {
    fn from(output: Output) -> Self {
        Self { output }
    }
}
