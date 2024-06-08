use crate::{assert_not_contains, handle_failed_output};
use std::ffi::OsStr;
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::process::{Command as StdCommand, ExitStatus, Output, Stdio};

/// This is a custom command wrapper that simplifies working with commands
/// and makes it easier to ensure that we check the exit status of executed
/// processes.
#[derive(Debug)]
pub struct Command {
    cmd: StdCommand,
    stdin: Option<Box<[u8]>>,
}

impl Command {
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        Self { cmd: StdCommand::new(program), stdin: None }
    }

    pub fn set_stdin(&mut self, stdin: Box<[u8]>) {
        self.stdin = Some(stdin);
    }

    /// Run the constructed command and assert that it is successfully run.
    #[track_caller]
    pub fn run(&mut self) -> CompletedProcess {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.command_output();
        if !output.status().success() {
            handle_failed_output(&self, output, caller_line_number);
        }
        output
    }

    /// Run the constructed command and assert that it does not successfully run.
    #[track_caller]
    pub fn run_fail(&mut self) -> CompletedProcess {
        let caller_location = std::panic::Location::caller();
        let caller_line_number = caller_location.line();

        let output = self.command_output();
        if output.status().success() {
            handle_failed_output(&self, output, caller_line_number);
        }
        output
    }

    #[track_caller]
    pub(crate) fn command_output(&mut self) -> CompletedProcess {
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

impl Deref for Command {
    type Target = StdCommand;

    fn deref(&self) -> &Self::Target {
        &self.cmd
    }
}

impl DerefMut for Command {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cmd
    }
}

/// Represents the result of an executed process.
/// The various `assert_` helper methods should preferably be used for
/// checking the contents of stdout/stderr.
pub struct CompletedProcess {
    output: Output,
}

impl CompletedProcess {
    pub fn stdout_utf8(&self) -> String {
        String::from_utf8(self.output.stdout.clone()).expect("stdout is not valid UTF-8")
    }

    pub fn stderr_utf8(&self) -> String {
        String::from_utf8(self.output.stderr.clone()).expect("stderr is not valid UTF-8")
    }

    pub fn status(&self) -> ExitStatus {
        self.output.status
    }

    /// Checks that trimmed `stdout` matches trimmed `content`.
    #[track_caller]
    pub fn assert_stdout_equals<S: AsRef<str>>(self, content: S) -> Self {
        assert_eq!(self.stdout_utf8().trim(), content.as_ref().trim());
        self
    }

    #[track_caller]
    pub fn assert_stdout_not_contains<S: AsRef<str>>(self, needle: S) -> Self {
        assert_not_contains(&self.stdout_utf8(), needle.as_ref());
        self
    }

    /// Checks that trimmed `stderr` matches trimmed `content`.
    #[track_caller]
    pub fn assert_stderr_equals<S: AsRef<str>>(self, content: S) -> Self {
        assert_eq!(self.stderr_utf8().trim(), content.as_ref().trim());
        self
    }

    #[track_caller]
    pub fn assert_stderr_contains<S: AsRef<str>>(self, needle: S) -> Self {
        assert!(self.stderr_utf8().contains(needle.as_ref()));
        self
    }

    #[track_caller]
    pub fn assert_stderr_not_contains<S: AsRef<str>>(self, needle: S) -> Self {
        assert_not_contains(&self.stdout_utf8(), needle.as_ref());
        self
    }

    #[track_caller]
    pub fn assert_exit_code(self, code: i32) -> Self {
        assert!(self.output.status.code() == Some(code));
        self
    }
}

impl From<Output> for CompletedProcess {
    fn from(output: Output) -> Self {
        Self { output }
    }
}
