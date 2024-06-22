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
    All,
    /// Print the output (by inheriting stdout/stderr).
    OnlyOutput,
    /// Suppress the output if the command succeeds, otherwise print the output.
    OnlyOnFailure,
}

/// Wrapper around `std::process::Command`.
#[derive(Debug)]
pub struct BootstrapCommand<'a> {
    pub command: &'a mut Command,
    pub failure_behavior: BehaviorOnFailure,
    pub output_mode: Option<OutputMode>,
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
        self.output_mode(OutputMode::OnlyOnFailure)
    }

    pub fn output_mode(self, output_mode: OutputMode) -> Self {
        Self { output_mode: Some(output_mode), ..self }
    }
}

impl<'a> From<&'a mut Command> for BootstrapCommand<'a> {
    fn from(command: &'a mut Command) -> Self {
        Self { command, failure_behavior: BehaviorOnFailure::Exit, output_mode: None }
    }
}

/// Represents the output of an executed process.
#[allow(unused)]
pub struct CommandOutput(Output);

impl CommandOutput {
    pub fn is_success(&self) -> bool {
        self.0.status.success()
    }

    pub fn is_failure(&self) -> bool {
        !self.is_success()
    }

    pub fn status(&self) -> ExitStatus {
        self.0.status
    }

    pub fn stdout(&self) -> String {
        String::from_utf8(self.0.stdout.clone()).expect("Cannot parse process stdout as UTF-8")
    }

    pub fn stderr(&self) -> String {
        String::from_utf8(self.0.stderr.clone()).expect("Cannot parse process stderr as UTF-8")
    }
}

impl Default for CommandOutput {
    fn default() -> Self {
        Self(Output { status: Default::default(), stdout: vec![], stderr: vec![] })
    }
}

impl From<Output> for CommandOutput {
    fn from(output: Output) -> Self {
        Self(output)
    }
}

impl From<ExitStatus> for CommandOutput {
    fn from(status: ExitStatus) -> Self {
        Self(Output { status, stdout: vec![], stderr: vec![] })
    }
}
