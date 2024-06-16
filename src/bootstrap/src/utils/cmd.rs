use std::ffi::OsStr;
use std::path::Path;
use std::process::{Command, ExitStatus, Output};

use crate::Context;

/// Wrapper around `std::process::Command` whose execution will be eventually
/// tracked.
#[derive(Debug)]
pub struct TrackedCommand {
    cmd: Command,
}

impl TrackedCommand {
    pub fn new<P: AsRef<OsStr>>(program: P) -> Self {
        Self { cmd: Command::new(program) }
    }

    pub fn current_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.cmd.current_dir(path);
        self
    }

    pub fn arg<A: AsRef<OsStr>>(mut self, arg: A) -> Self {
        self.cmd.arg(arg);
        self
    }

    /// Runs the command, ensuring that it has succeeded.
    #[track_caller]
    pub fn run(&mut self, ctx: &Context) -> CommandOutput {
        let output = self.run_maybe(ctx);
        if !output.is_success() {
            panic!(
                "`{:?}`\nwas supposed to succeed, but it failed with {}\nStdout: {}\nStderr: {}",
                self.cmd,
                output.status,
                output.stdout(),
                output.stderr()
            )
        }
        output
    }

    /// Runs the command.
    #[track_caller]
    pub fn run_maybe(&mut self, _ctx: &Context) -> CommandOutput {
        let output = self.cmd.output().expect("Cannot execute process");
        output.into()
    }

    /// Runs the command, ensuring that it has succeeded, and returns its stdout.
    #[track_caller]
    pub fn run_output(&mut self, ctx: &Context) -> String {
        self.run(ctx).stdout()
    }
}

/// Creates a new tracked command.
pub fn cmd<P: AsRef<OsStr>>(program: P) -> TrackedCommand {
    TrackedCommand::new(program)
}

/// Represents the output of an executed process.
#[allow(unused)]
pub struct CommandOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl CommandOutput {
    pub fn is_success(&self) -> bool {
        self.status.success()
    }

    pub fn is_failure(&self) -> bool {
        !self.is_success()
    }

    pub fn stdout(&self) -> String {
        String::from_utf8(self.stdout.clone()).expect("Cannot parse process stdout as UTF-8")
    }

    pub fn stderr(&self) -> String {
        String::from_utf8(self.stderr.clone()).expect("Cannot parse process stderr as UTF-8")
    }
}

impl From<Output> for CommandOutput {
    fn from(output: Output) -> Self {
        Self { status: output.status, stdout: output.stdout, stderr: output.stderr }
    }
}
