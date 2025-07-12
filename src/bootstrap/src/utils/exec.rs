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
use std::fs::File;
use std::hash::Hash;
use std::io::{BufWriter, Write};
use std::panic::Location;
use std::path::Path;
use std::process;
use std::process::{
    Child, ChildStderr, ChildStdout, Command, CommandArgs, CommandEnvs, ExitStatus, Output, Stdio,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
pub struct CommandFingerprint {
    program: OsString,
    args: Vec<OsString>,
    envs: Vec<(OsString, Option<OsString>)>,
    cwd: Option<PathBuf>,
}

impl CommandFingerprint {
    /// Helper method to format both Command and BootstrapCommand as a short execution line,
    /// without all the other details (e.g. environment variables).
    pub fn format_short_cmd(&self) -> String {
        let program = Path::new(&self.program);
        let mut line = vec![program.file_name().unwrap().to_str().unwrap().to_owned()];
        line.extend(self.args.iter().map(|arg| arg.to_string_lossy().into_owned()));
        line.extend(self.cwd.iter().map(|p| p.to_string_lossy().into_owned()));
        line.join(" ")
    }
}

#[derive(Default, Clone)]
pub struct CommandProfile {
    pub traces: Vec<ExecutionTrace>,
}

#[derive(Default)]
pub struct CommandProfiler {
    stats: Mutex<HashMap<CommandFingerprint, CommandProfile>>,
}

impl CommandProfiler {
    pub fn record_execution(&self, key: CommandFingerprint, start_time: Instant) {
        let mut stats = self.stats.lock().unwrap();
        let entry = stats.entry(key).or_default();
        entry.traces.push(ExecutionTrace::Executed { duration: start_time.elapsed() });
    }

    pub fn record_cache_hit(&self, key: CommandFingerprint) {
        let mut stats = self.stats.lock().unwrap();
        let entry = stats.entry(key).or_default();
        entry.traces.push(ExecutionTrace::CacheHit);
    }

    pub fn report_summary(&self, start_time: Instant) {
        let pid = process::id();
        let filename = format!("bootstrap-profile-{pid}.txt");

        let file = match File::create(&filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create profiler output file: {e}");
                return;
            }
        };

        let mut writer = BufWriter::new(file);
        let stats = self.stats.lock().unwrap();

        let mut entries: Vec<_> = stats
            .iter()
            .map(|(key, profile)| {
                let max_duration = profile
                    .traces
                    .iter()
                    .filter_map(|trace| match trace {
                        ExecutionTrace::Executed { duration, .. } => Some(*duration),
                        _ => None,
                    })
                    .max();

                (key, profile, max_duration)
            })
            .collect();

        entries.sort_by(|a, b| b.2.cmp(&a.2));

        let total_bootstrap_duration = start_time.elapsed();

        let total_fingerprints = entries.len();
        let mut total_cache_hits = 0;
        let mut total_execution_duration = Duration::ZERO;
        let mut total_saved_duration = Duration::ZERO;

        for (key, profile, max_duration) in &entries {
            writeln!(writer, "Command: {:?}", key.format_short_cmd()).unwrap();

            let mut hits = 0;
            let mut runs = 0;
            let mut command_total_duration = Duration::ZERO;

            for trace in &profile.traces {
                match trace {
                    ExecutionTrace::CacheHit => {
                        hits += 1;
                    }
                    ExecutionTrace::Executed { duration, .. } => {
                        runs += 1;
                        command_total_duration += *duration;
                    }
                }
            }

            total_cache_hits += hits;
            total_execution_duration += command_total_duration;
            // This makes sense only in our current setup, where:
            // - If caching is enabled, we record the timing for the initial execution,
            //   and all subsequent runs will be cache hits.
            // - If caching is disabled or unused, there will be no cache hits,
            //   and we'll record timings for all executions.
            total_saved_duration += command_total_duration * hits as u32;

            let command_vs_bootstrap = if total_bootstrap_duration > Duration::ZERO {
                100.0 * command_total_duration.as_secs_f64()
                    / total_bootstrap_duration.as_secs_f64()
            } else {
                0.0
            };

            let duration_str = match max_duration {
                Some(d) => format!("{d:.2?}"),
                None => "-".into(),
            };

            writeln!(
                writer,
                "Summary: {runs} run(s), {hits} hit(s), max_duration={duration_str} total_duration: {command_total_duration:.2?} ({command_vs_bootstrap:.2?}% of total)\n"
            )
            .unwrap();
        }

        let overhead_time = total_bootstrap_duration
            .checked_sub(total_execution_duration)
            .unwrap_or(Duration::ZERO);

        writeln!(writer, "\n=== Aggregated Summary ===").unwrap();
        writeln!(writer, "Total unique commands (fingerprints): {total_fingerprints}").unwrap();
        writeln!(writer, "Total time spent in command executions: {total_execution_duration:.2?}")
            .unwrap();
        writeln!(writer, "Total bootstrap time: {total_bootstrap_duration:.2?}").unwrap();
        writeln!(writer, "Time spent outside command executions: {overhead_time:.2?}").unwrap();
        writeln!(writer, "Total cache hits: {total_cache_hits}").unwrap();
        writeln!(writer, "Estimated time saved due to cache hits: {total_saved_duration:.2?}")
            .unwrap();

        println!("Command profiler report saved to {filename}");
    }
}

#[derive(Clone)]
pub enum ExecutionTrace {
    CacheHit,
    Executed { duration: Duration },
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

    /// Spawn the command in background, while capturing and returning stdout, and printing stderr.
    /// Returns None in dry-mode
    #[track_caller]
    pub fn stream_capture_stdout(
        &'a mut self,
        exec_ctx: impl AsRef<ExecutionContext>,
    ) -> Option<StreamingCommand> {
        exec_ctx.as_ref().stream(self, OutputMode::Capture, OutputMode::Print)
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

    pub fn fingerprint(&self) -> CommandFingerprint {
        let command = &self.command;
        CommandFingerprint {
            program: command.get_program().into(),
            args: command.get_args().map(OsStr::to_os_string).collect(),
            envs: command
                .get_envs()
                .map(|(k, v)| (k.to_os_string(), v.map(|val| val.to_os_string())))
                .collect(),
            cwd: command.get_current_dir().map(Path::to_path_buf),
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

#[derive(Clone, Default)]
pub struct ExecutionContext {
    dry_run: DryRun,
    verbose: u8,
    pub fail_fast: bool,
    delayed_failures: Arc<Mutex<Vec<String>>>,
    command_cache: Arc<CommandCache>,
    profiler: Arc<CommandProfiler>,
}

#[derive(Default)]
pub struct CommandCache {
    cache: Mutex<HashMap<CommandFingerprint, CommandOutput>>,
}

enum CommandState<'a> {
    Cached(CommandOutput),
    Deferred {
        process: Option<Result<Child, std::io::Error>>,
        command: &'a mut BootstrapCommand,
        stdout: OutputMode,
        stderr: OutputMode,
        executed_at: &'a Location<'a>,
        fingerprint: CommandFingerprint,
        start_time: Instant,
    },
}

pub struct StreamingCommand {
    child: Child,
    pub stdout: Option<ChildStdout>,
    pub stderr: Option<ChildStderr>,
    fingerprint: CommandFingerprint,
    start_time: Instant,
}

#[must_use]
pub struct DeferredCommand<'a> {
    state: CommandState<'a>,
}

impl CommandCache {
    pub fn get(&self, key: &CommandFingerprint) -> Option<CommandOutput> {
        self.cache.lock().unwrap().get(key).cloned()
    }

    pub fn insert(&self, key: CommandFingerprint, output: CommandOutput) {
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

    pub fn profiler(&self) -> &CommandProfiler {
        &self.profiler
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
        let fingerprint = command.fingerprint();

        if let Some(cached_output) = self.command_cache.get(&fingerprint) {
            command.mark_as_executed();
            self.verbose(|| println!("Cache hit: {command:?}"));
            self.profiler.record_cache_hit(fingerprint);
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
                    fingerprint,
                    start_time: Instant::now(),
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

        let start_time = Instant::now();

        let child = cmd.spawn();

        DeferredCommand {
            state: CommandState::Deferred {
                process: Some(child),
                command,
                stdout,
                stderr,
                executed_at,
                fingerprint,
                start_time,
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

    /// Spawns the command with configured stdout and stderr handling.
    ///
    /// Returns None if in dry-run mode or Panics if the command fails to spawn.
    pub fn stream(
        &self,
        command: &mut BootstrapCommand,
        stdout: OutputMode,
        stderr: OutputMode,
    ) -> Option<StreamingCommand> {
        command.mark_as_executed();
        if !command.run_in_dry_run && self.dry_run() {
            return None;
        }
        let start_time = Instant::now();
        let fingerprint = command.fingerprint();
        let cmd = &mut command.command;
        cmd.stdout(stdout.stdio());
        cmd.stderr(stderr.stdio());
        let child = cmd.spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(e) => panic!("failed to execute command: {cmd:?}\nERROR: {e}"),
        };

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        Some(StreamingCommand { child, stdout, stderr, fingerprint, start_time })
    }
}

impl AsRef<ExecutionContext> for ExecutionContext {
    fn as_ref(&self) -> &ExecutionContext {
        self
    }
}

impl StreamingCommand {
    pub fn wait(
        mut self,
        exec_ctx: impl AsRef<ExecutionContext>,
    ) -> Result<ExitStatus, std::io::Error> {
        let exec_ctx = exec_ctx.as_ref();
        let output = self.child.wait();
        exec_ctx.profiler().record_execution(self.fingerprint, self.start_time);
        output
    }
}

impl<'a> DeferredCommand<'a> {
    pub fn wait_for_output(self, exec_ctx: impl AsRef<ExecutionContext>) -> CommandOutput {
        match self.state {
            CommandState::Cached(output) => output,
            CommandState::Deferred {
                process,
                command,
                stdout,
                stderr,
                executed_at,
                fingerprint,
                start_time,
            } => {
                let exec_ctx = exec_ctx.as_ref();

                let output =
                    Self::finish_process(process, command, stdout, stderr, executed_at, exec_ctx);

                if (!exec_ctx.dry_run() || command.run_in_dry_run)
                    && output.status().is_some()
                    && command.should_cache
                {
                    exec_ctx.command_cache.insert(fingerprint.clone(), output.clone());
                    exec_ctx.profiler.record_execution(fingerprint.clone(), start_time);
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
