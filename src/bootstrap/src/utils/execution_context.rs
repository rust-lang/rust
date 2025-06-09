//! Shared execution context for running bootstrap commands.
//!
//! This module provides the [`ExecutionContext`] type, which holds global configuration
//! relevant during the execution of commands in bootstrap. This includes dry-run
//! mode, verbosity level, and behavior on failure.
use std::sync::{Arc, Mutex};

use crate::core::config::DryRun;
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
    pub fn run(
        &self,
        command: &mut BootstrapCommand,
        stdout: OutputMode,
        stderr: OutputMode,
    ) -> CommandOutput {
        command.mark_as_executed();
        if self.dry_run() && !command.run_always {
            return CommandOutput::default();
        }

        #[cfg(feature = "tracing")]
        let _run_span = trace_cmd!(command);

        let created_at = command.get_created_location();
        let executed_at = std::panic::Location::caller();

        self.verbose(|| {
            println!("running: {command:?} (created at {created_at}, executed at {executed_at})")
        });

        let cmd = command.as_command_mut();
        cmd.stdout(stdout.stdio());
        cmd.stderr(stderr.stdio());

        let output = cmd.output();

        use std::fmt::Write;

        let mut message = String::new();
        let output: CommandOutput = match output {
            // Command has succeeded
            Ok(output) if output.status.success() => {
                CommandOutput::from_output(output, stdout, stderr)
            }
            // Command has started, but then it failed
            Ok(output) => {
                writeln!(
                    message,
                    r#"
Command {command:?} did not execute successfully.
Expected success, got {}
Created at: {created_at}
Executed at: {executed_at}"#,
                    output.status,
                )
                .unwrap();

                let output: CommandOutput = CommandOutput::from_output(output, stdout, stderr);

                // If the output mode is OutputMode::Capture, we can now print the output.
                // If it is OutputMode::Print, then the output has already been printed to
                // stdout/stderr, and we thus don't have anything captured to print anyway.
                if stdout.captures() {
                    writeln!(message, "\nSTDOUT ----\n{}", output.stdout().trim()).unwrap();
                }
                if stderr.captures() {
                    writeln!(message, "\nSTDERR ----\n{}", output.stderr().trim()).unwrap();
                }
                output
            }
            // The command did not even start
            Err(e) => {
                writeln!(
                    message,
                    "\n\nCommand {command:?} did not execute successfully.\
            \nIt was not possible to execute the command: {e:?}"
                )
                .unwrap();
                CommandOutput::did_not_start(stdout, stderr)
            }
        };

        let fail = |message: &str, output: CommandOutput| -> ! {
            if self.is_verbose() {
                println!("{message}");
            } else {
                let (stdout, stderr) = (output.stdout_if_present(), output.stderr_if_present());
                // If the command captures output, the user would not see any indication that
                // it has failed. In this case, print a more verbose error, since to provide more
                // context.
                if stdout.is_some() || stderr.is_some() {
                    if let Some(stdout) =
                        output.stdout_if_present().take_if(|s| !s.trim().is_empty())
                    {
                        println!("STDOUT:\n{stdout}\n");
                    }
                    if let Some(stderr) =
                        output.stderr_if_present().take_if(|s| !s.trim().is_empty())
                    {
                        println!("STDERR:\n{stderr}\n");
                    }
                    println!("Command {command:?} has failed. Rerun with -v to see more details.");
                } else {
                    println!("Command has failed. Rerun with -v to see more details.");
                }
            }
            exit!(1);
        };

        if !output.is_success() {
            match command.failure_behavior {
                BehaviorOnFailure::DelayFail => {
                    if self.fail_fast {
                        fail(&message, output);
                    }

                    self.add_to_delay_failure(message);
                }
                BehaviorOnFailure::Exit => {
                    fail(&message, output);
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
