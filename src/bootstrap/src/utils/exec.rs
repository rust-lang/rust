use std::process::Command;

/// What should be done when the command fails.
#[derive(Debug, Copy, Clone)]
pub enum BehaviorOnFailure {
    /// Immediately stop bootstrap.
    Exit,
    /// Delay failure until the end of bootstrap invocation.
    DelayFail,
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
    Suppress,
}

/// Wrapper around `std::process::Command`.
#[derive(Debug)]
pub struct BootstrapCommand<'a> {
    pub command: &'a mut Command,
    pub failure_behavior: Option<BehaviorOnFailure>,
    pub output_mode: OutputMode,
}

impl<'a> BootstrapCommand<'a> {
    pub fn delay_failure(self) -> Self {
        Self { failure_behavior: Some(BehaviorOnFailure::DelayFail), ..self }
    }
    pub fn fail_fast(self) -> Self {
        Self { failure_behavior: Some(BehaviorOnFailure::Exit), ..self }
    }
    pub fn output_mode(self, output_mode: OutputMode) -> Self {
        Self { output_mode, ..self }
    }
}

impl<'a> From<&'a mut Command> for BootstrapCommand<'a> {
    fn from(command: &'a mut Command) -> Self {
        Self { command, failure_behavior: None, output_mode: OutputMode::Suppress }
    }
}
