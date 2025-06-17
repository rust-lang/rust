//! Shared execution context for running bootstrap commands.
//!
//! This module provides the [`ExecutionContext`] type, which holds global configuration
//! relevant during the execution of commands in bootstrap. This includes dry-run
//! mode, verbosity level, and behavior on failure.
use std::panic::Location;
use std::process::Child;
use std::sync::{Arc, Mutex};

use crate::core::config::DryRun;
#[cfg(feature = "tracing")]
use crate::trace_cmd;
use crate::{BehaviorOnFailure, BootstrapCommand, CommandOutput, OutputMode, exit};

#[derive(Clone, Default)]
pub struct ExecutionContext {
    dry_run: DryRun,
    verbose: u8,
    pub fail_fast: bool,
    delayed_failures: Arc<Mutex<Vec<String>>>,
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
        command.mark_as_executed();

        let created_at = command.get_created_location();
        let executed_at = std::panic::Location::caller();

        if self.dry_run() && !command.run_always {
            return DeferredCommand { process: None, stdout, stderr, command, executed_at };
        }

        #[cfg(feature = "tracing")]
        let _run_span = trace_cmd!(command);

        self.verbose(|| {
            println!("running: {command:?} (created at {created_at}, executed at {executed_at})")
        });

        let cmd = command.as_command_mut();
        cmd.stdout(stdout.stdio());
        cmd.stderr(stderr.stdio());

        let child = cmd.spawn();

        DeferredCommand { process: Some(child), stdout, stderr, command, executed_at }
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

pub struct DeferredCommand<'a> {
    process: Option<Result<Child, std::io::Error>>,
    command: &'a mut BootstrapCommand,
    stdout: OutputMode,
    stderr: OutputMode,
    executed_at: &'a Location<'a>,
}

impl<'a> DeferredCommand<'a> {
    pub fn wait_for_output(mut self, exec_ctx: impl AsRef<ExecutionContext>) -> CommandOutput {
        let exec_ctx = exec_ctx.as_ref();

        let process = match self.process.take() {
            Some(p) => p,
            None => return CommandOutput::default(),
        };

        let created_at = self.command.get_created_location();
        let executed_at = self.executed_at;

        let mut message = String::new();

        let output = match process {
            Ok(child) => match child.wait_with_output() {
                Ok(result) if result.status.success() => {
                    // Successful execution
                    CommandOutput::from_output(result, self.stdout, self.stderr)
                }
                Ok(result) => {
                    // Command ran but failed
                    use std::fmt::Write;

                    writeln!(
                        message,
                        r#"
Command {:?} did not execute successfully.
Expected success, got {}
Created at: {created_at}
Executed at: {executed_at}"#,
                        self.command, result.status,
                    )
                    .unwrap();

                    let output = CommandOutput::from_output(result, self.stdout, self.stderr);

                    if self.stdout.captures() {
                        writeln!(message, "\nSTDOUT ----\n{}", output.stdout().trim()).unwrap();
                    }
                    if self.stderr.captures() {
                        writeln!(message, "\nSTDERR ----\n{}", output.stderr().trim()).unwrap();
                    }

                    output
                }
                Err(e) => {
                    // Failed to wait for output
                    use std::fmt::Write;

                    writeln!(
                        message,
                        "\n\nCommand {:?} did not execute successfully.\
                        \nIt was not possible to execute the command: {e:?}",
                        self.command
                    )
                    .unwrap();

                    CommandOutput::did_not_start(self.stdout, self.stderr)
                }
            },
            Err(e) => {
                // Failed to spawn the command
                use std::fmt::Write;

                writeln!(
                    message,
                    "\n\nCommand {:?} did not execute successfully.\
                    \nIt was not possible to execute the command: {e:?}",
                    self.command
                )
                .unwrap();

                CommandOutput::did_not_start(self.stdout, self.stderr)
            }
        };

        if !output.is_success() {
            match self.command.failure_behavior {
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
