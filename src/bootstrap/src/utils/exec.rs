//! Command Execution Module
//!
//! This module provides a structured way to execute and manage commands efficiently,
//! ensuring controlled failure handling and output management.

use std::ffi::OsStr;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::process::{Command, CommandArgs, CommandEnvs, ExitStatus, Output, Stdio};

use build_helper::ci::CiEnv;
use build_helper::drop_bomb::DropBomb;

use super::execution_context::ExecutionContext;

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

/// How should the output of a specific stream of the command (stdout/stderr) be handled
/// (whether it should be captured or printed).
#[derive(Debug, Copy, Clone)]
pub enum OutputMode {
    /// Prints the stream by inheriting it from the bootstrap process.
    Print,
    /// Captures the stream into memory.
    Capture,
}

impl OutputMode {
    pub fn captures(&self) -> bool {
        match self {
            OutputMode::Print => false,
            OutputMode::Capture => true,
        }
    }

    pub fn stdio(&self) -> Stdio {
        match self {
            OutputMode::Print => Stdio::inherit(),
            OutputMode::Capture => Stdio::piped(),
        }
    }
}

/// Wrapper around `std::process::Command`.
///
/// By default, the command will exit bootstrap if it fails.
/// If you want to allow failures, use [allow_failure].
/// If you want to delay failures until the end of bootstrap, use [delay_failure].
///
/// By default, the command will print its stdout/stderr to stdout/stderr of bootstrap ([OutputMode::Print]).
/// If you want to handle the output programmatically, use [BootstrapCommand::run_capture].
///
/// Bootstrap will print a debug log to stdout if the command fails and failure is not allowed.
///
/// [allow_failure]: BootstrapCommand::allow_failure
/// [delay_failure]: BootstrapCommand::delay_failure
pub struct BootstrapCommand {
    command: Command,
    pub failure_behavior: BehaviorOnFailure,
    // Run the command even during dry run
    pub run_always: bool,
    // This field makes sure that each command is executed (or disarmed) before it is dropped,
    // to avoid forgetting to execute a command.
    drop_bomb: DropBomb,
}

impl BootstrapCommand {
    #[track_caller]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        Command::new(program).into()
    }

    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.command.arg(arg.as_ref());
        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.command.args(args);
        self
    }

    pub fn env<K, V>(&mut self, key: K, val: V) -> &mut Self
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.command.env(key, val);
        self
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.command.get_envs()
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        self.command.get_args()
    }

    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Self {
        self.command.env_remove(key);
        self
    }

    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Self {
        self.command.current_dir(dir);
        self
    }

    #[must_use]
    pub fn delay_failure(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::DelayFail, ..self }
    }

    pub fn fail_fast(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::Exit, ..self }
    }

    #[must_use]
    pub fn allow_failure(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::Ignore, ..self }
    }

    pub fn run_always(&mut self) -> &mut Self {
        self.run_always = true;
        self
    }

    /// Run the command, while printing stdout and stderr.
    /// Returns true if the command has succeeded.
    #[track_caller]
    pub fn run(&mut self, exec_ctx: impl AsRef<ExecutionContext>) -> bool {
        exec_ctx.as_ref().run(self, OutputMode::Print, OutputMode::Print).is_success()
    }

    /// Run the command, while capturing and returning all its output.
    #[track_caller]
    pub fn run_capture(&mut self, exec_ctx: impl AsRef<ExecutionContext>) -> CommandOutput {
        exec_ctx.as_ref().run(self, OutputMode::Capture, OutputMode::Capture)
    }

    /// Run the command, while capturing and returning stdout, and printing stderr.
    #[track_caller]
    pub fn run_capture_stdout(&mut self, exec_ctx: impl AsRef<ExecutionContext>) -> CommandOutput {
        exec_ctx.as_ref().run(self, OutputMode::Capture, OutputMode::Print)
    }

    /// Provides access to the stdlib Command inside.
    /// FIXME: This function should be eventually removed from bootstrap.
    pub fn as_command_mut(&mut self) -> &mut Command {
        // We don't know what will happen with the returned command, so we need to mark this
        // command as executed proactively.
        self.mark_as_executed();
        &mut self.command
    }

    /// Mark the command as being executed, disarming the drop bomb.
    /// If this method is not called before the command is dropped, its drop will panic.
    pub fn mark_as_executed(&mut self) {
        self.drop_bomb.defuse();
    }

    /// Returns the source code location where this command was created.
    pub fn get_created_location(&self) -> std::panic::Location<'static> {
        self.drop_bomb.get_created_location()
    }

    /// If in a CI environment, forces the command to run with colors.
    pub fn force_coloring_in_ci(&mut self) {
        if CiEnv::is_ci() {
            // Due to use of stamp/docker, the output stream of bootstrap is not
            // a TTY in CI, so coloring is by-default turned off.
            // The explicit `TERM=xterm` environment is needed for
            // `--color always` to actually work. This env var was lost when
            // compiling through the Makefile. Very strange.
            self.env("TERM", "xterm").args(["--color", "always"]);
        }
    }
}

impl Debug for BootstrapCommand {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.command)?;
        write!(f, " (failure_mode={:?})", self.failure_behavior)
    }
}

impl From<Command> for BootstrapCommand {
    #[track_caller]
    fn from(command: Command) -> Self {
        let program = command.get_program().to_owned();

        Self {
            command,
            failure_behavior: BehaviorOnFailure::Exit,
            run_always: false,
            drop_bomb: DropBomb::arm(program),
        }
    }
}

/// Represents the current status of `BootstrapCommand`.
enum CommandStatus {
    /// The command has started and finished with some status.
    Finished(ExitStatus),
    /// It was not even possible to start the command.
    DidNotStart,
}

/// Create a new BootstrapCommand. This is a helper function to make command creation
/// shorter than `BootstrapCommand::new`.
#[track_caller]
#[must_use]
pub fn command<S: AsRef<OsStr>>(program: S) -> BootstrapCommand {
    BootstrapCommand::new(program)
}

/// Represents the output of an executed process.
pub struct CommandOutput {
    status: CommandStatus,
    stdout: Option<Vec<u8>>,
    stderr: Option<Vec<u8>>,
}

impl CommandOutput {
    #[must_use]
    pub fn did_not_start(stdout: OutputMode, stderr: OutputMode) -> Self {
        Self {
            status: CommandStatus::DidNotStart,
            stdout: match stdout {
                OutputMode::Print => None,
                OutputMode::Capture => Some(vec![]),
            },
            stderr: match stderr {
                OutputMode::Print => None,
                OutputMode::Capture => Some(vec![]),
            },
        }
    }

    #[must_use]
    pub fn from_output(output: Output, stdout: OutputMode, stderr: OutputMode) -> Self {
        Self {
            status: CommandStatus::Finished(output.status),
            stdout: match stdout {
                OutputMode::Print => None,
                OutputMode::Capture => Some(output.stdout),
            },
            stderr: match stderr {
                OutputMode::Print => None,
                OutputMode::Capture => Some(output.stderr),
            },
        }
    }

    #[must_use]
    pub fn is_success(&self) -> bool {
        match self.status {
            CommandStatus::Finished(status) => status.success(),
            CommandStatus::DidNotStart => false,
        }
    }

    #[must_use]
    pub fn is_failure(&self) -> bool {
        !self.is_success()
    }

    pub fn status(&self) -> Option<ExitStatus> {
        match self.status {
            CommandStatus::Finished(status) => Some(status),
            CommandStatus::DidNotStart => None,
        }
    }

    #[must_use]
    pub fn stdout(&self) -> String {
        String::from_utf8(
            self.stdout.clone().expect("Accessing stdout of a command that did not capture stdout"),
        )
        .expect("Cannot parse process stdout as UTF-8")
    }

    #[must_use]
    pub fn stdout_if_present(&self) -> Option<String> {
        self.stdout.as_ref().and_then(|s| String::from_utf8(s.clone()).ok())
    }

    #[must_use]
    pub fn stdout_if_ok(&self) -> Option<String> {
        if self.is_success() { Some(self.stdout()) } else { None }
    }

    #[must_use]
    pub fn stderr(&self) -> String {
        String::from_utf8(
            self.stderr.clone().expect("Accessing stderr of a command that did not capture stderr"),
        )
        .expect("Cannot parse process stderr as UTF-8")
    }

    #[must_use]
    pub fn stderr_if_present(&self) -> Option<String> {
        self.stderr.as_ref().and_then(|s| String::from_utf8(s.clone()).ok())
    }
}

impl Default for CommandOutput {
    fn default() -> Self {
        Self {
            status: CommandStatus::Finished(ExitStatus::default()),
            stdout: Some(vec![]),
            stderr: Some(vec![]),
        }
    }
}

/// Helper trait to format both Command and BootstrapCommand as a short execution line,
/// without all the other details (e.g. environment variables).
#[cfg(feature = "tracing")]
pub trait FormatShortCmd {
    fn format_short_cmd(&self) -> String;
}

#[cfg(feature = "tracing")]
impl FormatShortCmd for BootstrapCommand {
    fn format_short_cmd(&self) -> String {
        self.command.format_short_cmd()
    }
}

#[cfg(feature = "tracing")]
impl FormatShortCmd for Command {
    fn format_short_cmd(&self) -> String {
        let program = Path::new(self.get_program());
        let mut line = vec![program.file_name().unwrap().to_str().unwrap()];
        line.extend(self.get_args().map(|arg| arg.to_str().unwrap()));
        line.join(" ")
    }
}
