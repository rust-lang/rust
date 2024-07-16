use crate::Build;
use build_helper::ci::CiEnv;
use build_helper::drop_bomb::DropBomb;
use std::ffi::OsStr;
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::process::{Command, CommandArgs, CommandEnvs, ExitStatus, Output, Stdio};

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
/// If you want to handle the output programmatically, use [BootstrapCommand::capture].
///
/// Bootstrap will print a debug log to stdout if the command fails and failure is not allowed.
///
/// [allow_failure]: BootstrapCommand::allow_failure
/// [delay_failure]: BootstrapCommand::delay_failure
pub struct BootstrapCommand {
    command: Command,
    pub failure_behavior: BehaviorOnFailure,
    pub stdout: OutputMode,
    pub stderr: OutputMode,
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

    #[must_use]
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

    /// Capture all output of the command, do not print it.
    #[must_use]
    pub fn capture(self) -> Self {
        Self { stdout: OutputMode::Capture, stderr: OutputMode::Capture, ..self }
    }

    /// Capture stdout of the command, do not print it.
    #[must_use]
    pub fn capture_stdout(self) -> Self {
        Self { stdout: OutputMode::Capture, ..self }
    }

    /// Run the command, returning its output.
    #[track_caller]
    pub fn run(&mut self, builder: &Build) -> CommandOutput {
        builder.run(self)
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
    pub fn force_coloring_in_ci(&mut self, ci_env: CiEnv) {
        if ci_env != CiEnv::None {
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
        write!(
            f,
            " (failure_mode={:?}, stdout_mode={:?}, stderr_mode={:?})",
            self.failure_behavior, self.stdout, self.stderr
        )
    }
}

impl From<Command> for BootstrapCommand {
    #[track_caller]
    fn from(command: Command) -> Self {
        let program = command.get_program().to_owned();

        Self {
            command,
            failure_behavior: BehaviorOnFailure::Exit,
            stdout: OutputMode::Print,
            stderr: OutputMode::Print,
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
#[allow(unused)]
pub struct CommandOutput {
    status: CommandStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl CommandOutput {
    #[must_use]
    pub fn did_not_start() -> Self {
        Self { status: CommandStatus::DidNotStart, stdout: vec![], stderr: vec![] }
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

    #[must_use]
    pub fn status(&self) -> Option<ExitStatus> {
        match self.status {
            CommandStatus::Finished(status) => Some(status),
            CommandStatus::DidNotStart => None,
        }
    }

    #[must_use]
    pub fn stdout(&self) -> String {
        String::from_utf8(self.stdout.clone()).expect("Cannot parse process stdout as UTF-8")
    }

    #[must_use]
    pub fn stdout_if_ok(&self) -> Option<String> {
        if self.is_success() { Some(self.stdout()) } else { None }
    }

    #[must_use]
    pub fn stderr(&self) -> String {
        String::from_utf8(self.stderr.clone()).expect("Cannot parse process stderr as UTF-8")
    }
}

impl Default for CommandOutput {
    fn default() -> Self {
        Self {
            status: CommandStatus::Finished(ExitStatus::default()),
            stdout: vec![],
            stderr: vec![],
        }
    }
}

impl From<Output> for CommandOutput {
    fn from(output: Output) -> Self {
        Self {
            status: CommandStatus::Finished(output.status),
            stdout: output.stdout,
            stderr: output.stderr,
        }
    }
}
