use std::ffi::OsStr;
use std::path::Path;
use std::process::{Command, CommandArgs, CommandEnvs, ExitStatus, Output};

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

/// How should the output of the command be handled (whether it should be captured or printed).
#[derive(Debug, Copy, Clone)]
pub enum OutputMode {
    /// Prints the stdout/stderr of the command to stdout/stderr of bootstrap (by inheriting these
    /// streams).
    /// Corresponds to calling `cmd.status()`.
    Print,
    /// Captures the stdout and stderr of the command into memory.
    /// Corresponds to calling `cmd.output()`.
    CaptureAll,
    /// Captures the stdout of the command into memory, inherits stderr.
    /// Corresponds to calling `cmd.output()`.
    CaptureStdout,
}

/// Wrapper around `std::process::Command`.
///
/// By default, the command will exit bootstrap if it fails.
/// If you want to allow failures, use [allow_failure].
/// If you want to delay failures until the end of bootstrap, use [delay_failure].
///
/// By default, the command will print its stdout/stderr to stdout/stderr of bootstrap
/// ([OutputMode::Print]).
/// If you want to handle the output programmatically, use [BootstrapCommand::capture].
///
/// Bootstrap will print a debug log to stdout if the command fails and failure is not allowed.
///
/// [allow_failure]: BootstrapCommand::allow_failure
/// [delay_failure]: BootstrapCommand::delay_failure
#[derive(Debug)]
pub struct BootstrapCommand {
    pub command: Command,
    pub failure_behavior: BehaviorOnFailure,
    pub output_mode: OutputMode,
}

impl BootstrapCommand {
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

    pub fn delay_failure(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::DelayFail, ..self }
    }

    pub fn fail_fast(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::Exit, ..self }
    }

    pub fn allow_failure(self) -> Self {
        Self { failure_behavior: BehaviorOnFailure::Ignore, ..self }
    }

    /// Capture the output of the command, do not print it.
    pub fn capture(self) -> Self {
        Self { output_mode: OutputMode::CaptureAll, ..self }
    }

    /// Capture stdout of the command, do not print it.
    pub fn capture_stdout(self) -> Self {
        Self { output_mode: OutputMode::CaptureStdout, ..self }
    }
}

/// This implementation exists to make it possible to pass both [BootstrapCommand] and
/// `&mut BootstrapCommand` to `Build.run()`.
impl AsMut<BootstrapCommand> for BootstrapCommand {
    fn as_mut(&mut self) -> &mut BootstrapCommand {
        self
    }
}

impl From<Command> for BootstrapCommand {
    fn from(command: Command) -> Self {
        Self { command, failure_behavior: BehaviorOnFailure::Exit, output_mode: OutputMode::Print }
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

    pub fn stdout_if_ok(&self) -> Option<String> {
        if self.is_success() { Some(self.stdout()) } else { None }
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
