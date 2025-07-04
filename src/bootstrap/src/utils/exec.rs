//! Command Execution Module
//!
//! Provides a structured interface for executing and managing commands during bootstrap,
//! with support for controlled failure handling and output management.
//!
//! This module defines the [`ExecutionContext`] type, which encapsulates global configuration
//! relevant to command execution in the bootstrap process. This includes settings such as
//! dry-run mode, verbosity level, and failure behavior.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::panic::Location;
use std::path::Path;
use std::process::{Child, Command, CommandArgs, CommandEnvs, ExitStatus, Output, Stdio};
use std::sync::{Arc, Mutex};

use build_helper::ci::CiEnv;
use build_helper::drop_bomb::DropBomb;
use build_helper::exit;

use crate::PathBuf;
use crate::core::config::DryRun;
#[cfg(feature = "tracing")]
use crate::trace_cmd;

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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct CommandCacheKey {
    program: OsString,
    args: Vec<OsString>,
    envs: Vec<(OsString, Option<OsString>)>,
    cwd: Option<PathBuf>,
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
/// By default, command executions are cached based on their workdir, program, arguments, and environment variables.
/// This avoids re-running identical commands unnecessarily, unless caching is explicitly disabled.
///
/// [allow_failure]: BootstrapCommand::allow_failure
/// [delay_failure]: BootstrapCommand::delay_failure
pub struct BootstrapCommand {
    command: Command,
    pub failure_behavior: BehaviorOnFailure,
    // Run the command even during dry run
    pub run_in_dry_run: bool,
    // This field makes sure that each command is executed (or disarmed) before it is dropped,
    // to avoid forgetting to execute a command.
    drop_bomb: DropBomb,
    should_cache: bool,
}

impl<'a> BootstrapCommand {
    #[track_caller]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        Command::new(program).into()
    }
    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.command.arg(arg.as_ref());
        self
    }

    pub fn do_not_cache(&mut self) -> &mut Self {
        self.should_cache = false;
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

    pub fn stdin(&mut self, stdin: std::process::Stdio) -> &mut Self {
        self.command.stdin(stdin);
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

    pub fn run_in_dry_run(&mut self) -> &mut Self {
        self.run_in_dry_run = true;
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

    /// Spawn the command in background, while capturing and returning all its output.
    #[track_caller]
    pub fn start_capture(
        &'a mut self,
        exec_ctx: impl AsRef<ExecutionContext>,
    ) -> DeferredCommand<'a> {
        exec_ctx.as_ref().start(self, OutputMode::Capture, OutputMode::Capture)
    }

    /// Spawn the command in background, while capturing and returning stdout, and printing stderr.
    #[track_caller]
    pub fn start_capture_stdout(
        &'a mut self,
        exec_ctx: impl AsRef<ExecutionContext>,
    ) -> DeferredCommand<'a> {
        exec_ctx.as_ref().start(self, OutputMode::Capture, OutputMode::Print)
    }

    /// Provides access to the stdlib Command inside.
    /// FIXME: This function should be eventually removed from bootstrap.
    pub fn as_command_mut(&mut self) -> &mut Command {
        // We proactively mark this command as executed since we can't be certain how the returned
        // command will be handled. Caching must also be avoided here, as the inner command could be
        // modified externally without us being aware.
        self.mark_as_executed();
        self.do_not_cache();
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

    pub fn cache_key(&self) -> Option<CommandCacheKey> {
        if !self.should_cache {
            return None;
        }
        let command = &self.command;
        Some(CommandCacheKey {
            program: command.get_program().into(),
            args: command.get_args().map(OsStr::to_os_string).collect(),
            envs: command
                .get_envs()
                .map(|(k, v)| (k.to_os_string(), v.map(|val| val.to_os_string())))
                .collect(),
            cwd: command.get_current_dir().map(Path::to_path_buf),
        })
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
            should_cache: true,
            command,
            failure_behavior: BehaviorOnFailure::Exit,
            run_in_dry_run: false,
            drop_bomb: DropBomb::arm(program),
        }
    }
}

/// Represents the current status of `BootstrapCommand`.
#[derive(Clone, PartialEq)]
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
#[derive(Clone, PartialEq)]
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

#[derive(Clone, Default)]
pub struct ExecutionContext {
    dry_run: DryRun,
    verbose: u8,
    pub fail_fast: bool,
    delayed_failures: Arc<Mutex<Vec<String>>>,
    command_cache: Arc<CommandCache>,
}

#[derive(Default)]
pub struct CommandCache {
    cache: Mutex<HashMap<CommandCacheKey, CommandOutput>>,
}

enum CommandState<'a> {
    Cached(CommandOutput),
    Deferred {
        process: Option<Result<Child, std::io::Error>>,
        command: &'a mut BootstrapCommand,
        stdout: OutputMode,
        stderr: OutputMode,
        executed_at: &'a Location<'a>,
        cache_key: Option<CommandCacheKey>,
    },
}

#[must_use]
pub struct DeferredCommand<'a> {
    state: CommandState<'a>,
}

impl CommandCache {
    pub fn get(&self, key: &CommandCacheKey) -> Option<CommandOutput> {
        self.cache.lock().unwrap().get(key).cloned()
    }

    pub fn insert(&self, key: CommandCacheKey, output: CommandOutput) {
        self.cache.lock().unwrap().insert(key, output);
    }
}

impl ExecutionContext {
    pub fn new() -> Self {
        ExecutionContext::default()
    }

    pub fn dry_run(&self) -> bool {
        match self.dry_run {
            DryRun::Disabled => false,
            DryRun::SelfCheck | DryRun::UserSelected => true,
        }
    }

    pub fn get_dry_run(&self) -> &DryRun {
        &self.dry_run
    }

    pub fn verbose(&self, f: impl Fn()) {
        if self.is_verbose() {
            f()
        }
    }

    pub fn is_verbose(&self) -> bool {
        self.verbose > 0
    }

    pub fn fail_fast(&self) -> bool {
        self.fail_fast
    }

    pub fn set_dry_run(&mut self, value: DryRun) {
        self.dry_run = value;
    }

    pub fn set_verbose(&mut self, value: u8) {
        self.verbose = value;
    }

    pub fn set_fail_fast(&mut self, value: bool) {
        self.fail_fast = value;
    }

    pub fn add_to_delay_failure(&self, message: String) {
        self.delayed_failures.lock().unwrap().push(message);
    }

    pub fn report_failures_and_exit(&self) {
        let failures = self.delayed_failures.lock().unwrap();
        if failures.is_empty() {
            return;
        }
        eprintln!("\n{} command(s) did not execute successfully:\n", failures.len());
        for failure in &*failures {
            eprintln!("  - {failure}");
        }
        exit!(1);
    }

    /// Execute a command and return its output.
    /// Note: Ideally, you should use one of the BootstrapCommand::run* functions to
    /// execute commands. They internally call this method.
    #[track_caller]
    pub fn start<'a>(
        &self,
        command: &'a mut BootstrapCommand,
        stdout: OutputMode,
        stderr: OutputMode,
    ) -> DeferredCommand<'a> {
        let cache_key = command.cache_key();

        if let Some(cached_output) = cache_key.as_ref().and_then(|key| self.command_cache.get(key))
        {
            command.mark_as_executed();
            self.verbose(|| println!("Cache hit: {command:?}"));
            return DeferredCommand { state: CommandState::Cached(cached_output) };
        }

        let created_at = command.get_created_location();
        let executed_at = std::panic::Location::caller();

        if self.dry_run() && !command.run_in_dry_run {
            return DeferredCommand {
                state: CommandState::Deferred {
                    process: None,
                    command,
                    stdout,
                    stderr,
                    executed_at,
                    cache_key,
                },
            };
        }

        #[cfg(feature = "tracing")]
        let _run_span = trace_cmd!(command);

        self.verbose(|| {
            println!("running: {command:?} (created at {created_at}, executed at {executed_at})")
        });

        let cmd = &mut command.command;
        cmd.stdout(stdout.stdio());
        cmd.stderr(stderr.stdio());

        let child = cmd.spawn();

        DeferredCommand {
            state: CommandState::Deferred {
                process: Some(child),
                command,
                stdout,
                stderr,
                executed_at,
                cache_key,
            },
        }
    }

    /// Execute a command and return its output.
    /// Note: Ideally, you should use one of the BootstrapCommand::run* functions to
    /// execute commands. They internally call this method.
    #[track_caller]
    pub fn run(
        &self,
        command: &mut BootstrapCommand,
        stdout: OutputMode,
        stderr: OutputMode,
    ) -> CommandOutput {
        self.start(command, stdout, stderr).wait_for_output(self)
    }

    fn fail(&self, message: &str, output: CommandOutput) -> ! {
        if self.is_verbose() {
            println!("{message}");
        } else {
            let (stdout, stderr) = (output.stdout_if_present(), output.stderr_if_present());
            // If the command captures output, the user would not see any indication that
            // it has failed. In this case, print a more verbose error, since to provide more
            // context.
            if stdout.is_some() || stderr.is_some() {
                if let Some(stdout) = output.stdout_if_present().take_if(|s| !s.trim().is_empty()) {
                    println!("STDOUT:\n{stdout}\n");
                }
                if let Some(stderr) = output.stderr_if_present().take_if(|s| !s.trim().is_empty()) {
                    println!("STDERR:\n{stderr}\n");
                }
                println!("Command has failed. Rerun with -v to see more details.");
            } else {
                println!("Command has failed. Rerun with -v to see more details.");
            }
        }
        exit!(1);
    }
}

impl AsRef<ExecutionContext> for ExecutionContext {
    fn as_ref(&self) -> &ExecutionContext {
        self
    }
}

impl<'a> DeferredCommand<'a> {
    pub fn wait_for_output(self, exec_ctx: impl AsRef<ExecutionContext>) -> CommandOutput {
        match self.state {
            CommandState::Cached(output) => output,
            CommandState::Deferred { process, command, stdout, stderr, executed_at, cache_key } => {
                let exec_ctx = exec_ctx.as_ref();

                let output =
                    Self::finish_process(process, command, stdout, stderr, executed_at, exec_ctx);

                if (!exec_ctx.dry_run() || command.run_in_dry_run)
                    && let (Some(cache_key), Some(_)) = (&cache_key, output.status())
                {
                    exec_ctx.command_cache.insert(cache_key.clone(), output.clone());
                }

                output
            }
        }
    }

    pub fn finish_process(
        mut process: Option<Result<Child, std::io::Error>>,
        command: &mut BootstrapCommand,
        stdout: OutputMode,
        stderr: OutputMode,
        executed_at: &'a std::panic::Location<'a>,
        exec_ctx: &ExecutionContext,
    ) -> CommandOutput {
        command.mark_as_executed();

        let process = match process.take() {
            Some(p) => p,
            None => return CommandOutput::default(),
        };

        let created_at = command.get_created_location();

        let mut message = String::new();

        let output = match process {
            Ok(child) => match child.wait_with_output() {
                Ok(result) if result.status.success() => {
                    // Successful execution
                    CommandOutput::from_output(result, stdout, stderr)
                }
                Ok(result) => {
                    // Command ran but failed
                    use std::fmt::Write;

                    writeln!(
                        message,
                        r#"
Command {command:?} did not execute successfully.
Expected success, got {}
Created at: {created_at}
Executed at: {executed_at}"#,
                        result.status,
                    )
                    .unwrap();

                    let output = CommandOutput::from_output(result, stdout, stderr);

                    if stdout.captures() {
                        writeln!(message, "\nSTDOUT ----\n{}", output.stdout().trim()).unwrap();
                    }
                    if stderr.captures() {
                        writeln!(message, "\nSTDERR ----\n{}", output.stderr().trim()).unwrap();
                    }

                    output
                }
                Err(e) => {
                    // Failed to wait for output
                    use std::fmt::Write;

                    writeln!(
                        message,
                        "\n\nCommand {command:?} did not execute successfully.\
                        \nIt was not possible to execute the command: {e:?}"
                    )
                    .unwrap();

                    CommandOutput::did_not_start(stdout, stderr)
                }
            },
            Err(e) => {
                // Failed to spawn the command
                use std::fmt::Write;

                writeln!(
                    message,
                    "\n\nCommand {command:?} did not execute successfully.\
                    \nIt was not possible to execute the command: {e:?}"
                )
                .unwrap();

                CommandOutput::did_not_start(stdout, stderr)
            }
        };

        if !output.is_success() {
            match command.failure_behavior {
                BehaviorOnFailure::DelayFail => {
                    if exec_ctx.fail_fast {
                        exec_ctx.fail(&message, output);
                    }
                    exec_ctx.add_to_delay_failure(message);
                }
                BehaviorOnFailure::Exit => {
                    exec_ctx.fail(&message, output);
                }
                BehaviorOnFailure::Ignore => {
                    // If failures are allowed, either the error has been printed already
                    // (OutputMode::Print) or the user used a capture output mode and wants to
                    // handle the error output on their own.
                }
            }
        }

        output
    }
}
