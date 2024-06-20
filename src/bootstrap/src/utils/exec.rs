use std::process::{Command, ExitStatus, Output};

/// What should be done when the command fails.
#[derive(Debug, Copy, Clone)]
pub enum BehaviorOnFailure {
    /// Immediately stop bootstrap.
    Exit,
    /// Delay failure until the end of bootstrap invocation.
    DelayFail,
    /// Ignore the failure, the command can fail in an expected way.
    Ignore,
}

/// How should the output of the command be handled.
#[derive(Debug, Copy, Clone)]
pub enum OutputMode {
    /// Print both the output (by inheriting stdout/stderr) and also the command itself, if it
    /// fails.
    PrintAll,
    /// Print the output (by inheriting stdout/stderr).
    PrintOutput,
    /// Suppress the output if the command succeeds, otherwise print the output.
    PrintOnFailure,
}

/// Wrapper around `std::process::Command`.
#[derive(Debug)]
pub struct BootstrapCommand<'a> {
    pub command: &'a mut Command,
    pub failure_behavior: BehaviorOnFailure,
    pub output_mode: OutputMode,
}

impl<'a> BootstrapCommand<'a> {
    pub fn delay_failure(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::DelayFail, ..self }
    }

    pub fn fail_fast(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::Exit, ..self }
    }

    pub fn allow_failure(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::Ignore, ..self }
    }

    /// Do not print the output of the command, unless it fails.
    pub fn quiet(self) -> Self {
        self.output_mode(OutputMode::PrintOnFailure)
    }

    pub fn output_mode(self, output_mode: OutputMode) -> Self {
        Self { output_mode, ..self }
    }
}

impl<'a> From<&'a mut Command> for BootstrapCommand<'a> {
    fn from(command: &'a mut Command) -> Self {
        Self {
            command,
            failure_behavior: BehaviorOnFailure::Exit,
            output_mode: OutputMode::PrintAll,
        }
    }
}

/// Represents the output of an executed process.
#[allow(unused)]
#[derive(Default)]
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

    pub fn status(&self) -> ExitStatus {
        self.status
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

impl From<ExitStatus> for CommandOutput {
    fn from(status: ExitStatus) -> Self {
        Self { status, stdout: Default::default(), stderr: Default::default() }
    }
}
