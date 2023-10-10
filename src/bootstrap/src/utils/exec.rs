use std::process::Command;

/// What should be done when the command fails.
#[derive(Debug, Copy, Clone)]
pub enum BehaviorOnFailure {
    /// Immediately stop bootstrap.
    Exit,
    /// Delay failure until the end of bootstrap invocation.
    DelayFail,
}

/// Wrapper around `std::process::Command`.
#[derive(Debug)]
pub struct BootstrapCommand<'a> {
    pub command: &'a mut Command,
    pub failure_behavior: Option<BehaviorOnFailure>,
}

impl<'a> BootstrapCommand<'a> {
    pub fn delay_failure(self) -> Self {
        Self { failure_behavior: Some(BehaviorOnFailure::DelayFail), ..self }
    }
    pub fn fail_fast(self) -> Self {
        Self { failure_behavior: Some(BehaviorOnFailure::Exit), ..self }
    }
}

impl<'a> From<&'a mut Command> for BootstrapCommand<'a> {
    fn from(command: &'a mut Command) -> Self {
        Self { command, failure_behavior: None }
    }
}
