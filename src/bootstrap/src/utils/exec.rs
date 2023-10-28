use std::process::Command;

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
    SuppressOnSuccess,
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
