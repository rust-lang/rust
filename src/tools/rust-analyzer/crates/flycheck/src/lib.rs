//! Flycheck provides the functionality needed to run `cargo check` or
//! another compatible command (f.x. clippy) in a background thread and provide
//! LSP diagnostics based on the output of the command.

// FIXME: This crate now handles running `cargo test` needed in the test explorer in
// addition to `cargo check`. Either split it into 3 crates (one for test, one for check
// and one common utilities) or change its name and docs to reflect the current state.

#![warn(rust_2018_idioms, unused_lifetimes)]

use std::{fmt, io, path::PathBuf, process::Command, time::Duration};

use crossbeam_channel::{never, select, unbounded, Receiver, Sender};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use serde::Deserialize;

pub use cargo_metadata::diagnostic::{
    Applicability, Diagnostic, DiagnosticCode, DiagnosticLevel, DiagnosticSpan,
    DiagnosticSpanMacroExpansion,
};
use toolchain::Tool;

mod command;
mod test_runner;

use command::{CommandHandle, ParseFromLine};
pub use test_runner::{CargoTestHandle, CargoTestMessage, TestState};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum InvocationStrategy {
    Once,
    #[default]
    PerWorkspace,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InvocationLocation {
    Root(AbsPathBuf),
    #[default]
    Workspace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlycheckConfig {
    CargoCommand {
        command: String,
        target_triples: Vec<String>,
        all_targets: bool,
        no_default_features: bool,
        all_features: bool,
        features: Vec<String>,
        extra_args: Vec<String>,
        extra_env: FxHashMap<String, String>,
        ansi_color_output: bool,
        target_dir: Option<PathBuf>,
    },
    CustomCommand {
        command: String,
        args: Vec<String>,
        extra_env: FxHashMap<String, String>,
        invocation_strategy: InvocationStrategy,
        invocation_location: InvocationLocation,
    },
}

impl fmt::Display for FlycheckConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlycheckConfig::CargoCommand { command, .. } => write!(f, "cargo {command}"),
            FlycheckConfig::CustomCommand { command, args, .. } => {
                write!(f, "{command} {}", args.join(" "))
            }
        }
    }
}

/// Flycheck wraps the shared state and communication machinery used for
/// running `cargo check` (or other compatible command) and providing
/// diagnostics based on the output.
/// The spawned thread is shut down when this struct is dropped.
#[derive(Debug)]
pub struct FlycheckHandle {
    // XXX: drop order is significant
    sender: Sender<StateChange>,
    _thread: stdx::thread::JoinHandle,
    id: usize,
}

impl FlycheckHandle {
    pub fn spawn(
        id: usize,
        sender: Box<dyn Fn(Message) + Send>,
        config: FlycheckConfig,
        sysroot_root: Option<AbsPathBuf>,
        workspace_root: AbsPathBuf,
    ) -> FlycheckHandle {
        let actor = FlycheckActor::new(id, sender, config, sysroot_root, workspace_root);
        let (sender, receiver) = unbounded::<StateChange>();
        let thread = stdx::thread::Builder::new(stdx::thread::ThreadIntent::Worker)
            .name("Flycheck".to_owned())
            .spawn(move || actor.run(receiver))
            .expect("failed to spawn thread");
        FlycheckHandle { id, sender, _thread: thread }
    }

    /// Schedule a re-start of the cargo check worker to do a workspace wide check.
    pub fn restart_workspace(&self, saved_file: Option<AbsPathBuf>) {
        self.sender.send(StateChange::Restart { package: None, saved_file }).unwrap();
    }

    /// Schedule a re-start of the cargo check worker to do a package wide check.
    pub fn restart_for_package(&self, package: String) {
        self.sender
            .send(StateChange::Restart { package: Some(package), saved_file: None })
            .unwrap();
    }

    /// Stop this cargo check worker.
    pub fn cancel(&self) {
        self.sender.send(StateChange::Cancel).unwrap();
    }

    pub fn id(&self) -> usize {
        self.id
    }
}

pub enum Message {
    /// Request adding a diagnostic with fixes included to a file
    AddDiagnostic { id: usize, workspace_root: AbsPathBuf, diagnostic: Diagnostic },

    /// Request check progress notification to client
    Progress {
        /// Flycheck instance ID
        id: usize,
        progress: Progress,
    },
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Message::AddDiagnostic { id, workspace_root, diagnostic } => f
                .debug_struct("AddDiagnostic")
                .field("id", id)
                .field("workspace_root", workspace_root)
                .field("diagnostic_code", &diagnostic.code.as_ref().map(|it| &it.code))
                .finish(),
            Message::Progress { id, progress } => {
                f.debug_struct("Progress").field("id", id).field("progress", progress).finish()
            }
        }
    }
}

#[derive(Debug)]
pub enum Progress {
    DidStart,
    DidCheckCrate(String),
    DidFinish(io::Result<()>),
    DidCancel,
    DidFailToRestart(String),
}

enum StateChange {
    Restart { package: Option<String>, saved_file: Option<AbsPathBuf> },
    Cancel,
}

/// A [`FlycheckActor`] is a single check instance of a workspace.
struct FlycheckActor {
    /// The workspace id of this flycheck instance.
    id: usize,
    sender: Box<dyn Fn(Message) + Send>,
    config: FlycheckConfig,
    /// Either the workspace root of the workspace we are flychecking,
    /// or the project root of the project.
    root: AbsPathBuf,
    sysroot_root: Option<AbsPathBuf>,
    /// CargoHandle exists to wrap around the communication needed to be able to
    /// run `cargo check` without blocking. Currently the Rust standard library
    /// doesn't provide a way to read sub-process output without blocking, so we
    /// have to wrap sub-processes output handling in a thread and pass messages
    /// back over a channel.
    command_handle: Option<CommandHandle<CargoCheckMessage>>,
}

enum Event {
    RequestStateChange(StateChange),
    CheckEvent(Option<CargoCheckMessage>),
}

const SAVED_FILE_PLACEHOLDER: &str = "$saved_file";

impl FlycheckActor {
    fn new(
        id: usize,
        sender: Box<dyn Fn(Message) + Send>,
        config: FlycheckConfig,
        sysroot_root: Option<AbsPathBuf>,
        workspace_root: AbsPathBuf,
    ) -> FlycheckActor {
        tracing::info!(%id, ?workspace_root, "Spawning flycheck");
        FlycheckActor {
            id,
            sender,
            config,
            sysroot_root,
            root: workspace_root,
            command_handle: None,
        }
    }

    fn report_progress(&self, progress: Progress) {
        self.send(Message::Progress { id: self.id, progress });
    }

    fn next_event(&self, inbox: &Receiver<StateChange>) -> Option<Event> {
        let check_chan = self.command_handle.as_ref().map(|cargo| &cargo.receiver);
        if let Ok(msg) = inbox.try_recv() {
            // give restarts a preference so check outputs don't block a restart or stop
            return Some(Event::RequestStateChange(msg));
        }
        select! {
            recv(inbox) -> msg => msg.ok().map(Event::RequestStateChange),
            recv(check_chan.unwrap_or(&never())) -> msg => Some(Event::CheckEvent(msg.ok())),
        }
    }

    fn run(mut self, inbox: Receiver<StateChange>) {
        'event: while let Some(event) = self.next_event(&inbox) {
            match event {
                Event::RequestStateChange(StateChange::Cancel) => {
                    tracing::debug!(flycheck_id = self.id, "flycheck cancelled");
                    self.cancel_check_process();
                }
                Event::RequestStateChange(StateChange::Restart { package, saved_file }) => {
                    // Cancel the previously spawned process
                    self.cancel_check_process();
                    while let Ok(restart) = inbox.recv_timeout(Duration::from_millis(50)) {
                        // restart chained with a stop, so just cancel
                        if let StateChange::Cancel = restart {
                            continue 'event;
                        }
                    }

                    let command =
                        match self.check_command(package.as_deref(), saved_file.as_deref()) {
                            Some(c) => c,
                            None => continue,
                        };
                    let formatted_command = format!("{:?}", command);

                    tracing::debug!(?command, "will restart flycheck");
                    match CommandHandle::spawn(command) {
                        Ok(command_handle) => {
                            tracing::debug!(command = formatted_command, "did  restart flycheck");
                            self.command_handle = Some(command_handle);
                            self.report_progress(Progress::DidStart);
                        }
                        Err(error) => {
                            self.report_progress(Progress::DidFailToRestart(format!(
                                "Failed to run the following command: {} error={}",
                                formatted_command, error
                            )));
                        }
                    }
                }
                Event::CheckEvent(None) => {
                    tracing::debug!(flycheck_id = self.id, "flycheck finished");

                    // Watcher finished
                    let command_handle = self.command_handle.take().unwrap();
                    let formatted_handle = format!("{:?}", command_handle);

                    let res = command_handle.join();
                    if res.is_err() {
                        tracing::error!(
                            "Flycheck failed to run the following command: {}",
                            formatted_handle
                        );
                    }
                    self.report_progress(Progress::DidFinish(res));
                }
                Event::CheckEvent(Some(message)) => match message {
                    CargoCheckMessage::CompilerArtifact(msg) => {
                        tracing::trace!(
                            flycheck_id = self.id,
                            artifact = msg.target.name,
                            "artifact received"
                        );
                        self.report_progress(Progress::DidCheckCrate(msg.target.name));
                    }

                    CargoCheckMessage::Diagnostic(msg) => {
                        tracing::trace!(
                            flycheck_id = self.id,
                            message = msg.message,
                            "diagnostic received"
                        );
                        self.send(Message::AddDiagnostic {
                            id: self.id,
                            workspace_root: self.root.clone(),
                            diagnostic: msg,
                        });
                    }
                },
            }
        }
        // If we rerun the thread, we need to discard the previous check results first
        self.cancel_check_process();
    }

    fn cancel_check_process(&mut self) {
        if let Some(command_handle) = self.command_handle.take() {
            tracing::debug!(
                command = ?command_handle,
                "did  cancel flycheck"
            );
            command_handle.cancel();
            self.report_progress(Progress::DidCancel);
        }
    }

    /// Construct a `Command` object for checking the user's code. If the user
    /// has specified a custom command with placeholders that we cannot fill,
    /// return None.
    fn check_command(
        &self,
        package: Option<&str>,
        saved_file: Option<&AbsPath>,
    ) -> Option<Command> {
        let (mut cmd, args) = match &self.config {
            FlycheckConfig::CargoCommand {
                command,
                target_triples,
                no_default_features,
                all_targets,
                all_features,
                extra_args,
                features,
                extra_env,
                ansi_color_output,
                target_dir,
            } => {
                let mut cmd = Command::new(Tool::Cargo.path());
                if let Some(sysroot_root) = &self.sysroot_root {
                    cmd.env("RUSTUP_TOOLCHAIN", AsRef::<std::path::Path>::as_ref(sysroot_root));
                }
                cmd.arg(command);
                cmd.current_dir(&self.root);

                match package {
                    Some(pkg) => cmd.arg("-p").arg(pkg),
                    None => cmd.arg("--workspace"),
                };

                cmd.arg(if *ansi_color_output {
                    "--message-format=json-diagnostic-rendered-ansi"
                } else {
                    "--message-format=json"
                });

                cmd.arg("--manifest-path");
                cmd.arg(self.root.join("Cargo.toml").as_os_str());

                for target in target_triples {
                    cmd.args(["--target", target.as_str()]);
                }
                if *all_targets {
                    cmd.arg("--all-targets");
                }
                if *all_features {
                    cmd.arg("--all-features");
                } else {
                    if *no_default_features {
                        cmd.arg("--no-default-features");
                    }
                    if !features.is_empty() {
                        cmd.arg("--features");
                        cmd.arg(features.join(" "));
                    }
                }
                if let Some(target_dir) = target_dir {
                    cmd.arg("--target-dir").arg(target_dir);
                }
                cmd.envs(extra_env);
                (cmd, extra_args.clone())
            }
            FlycheckConfig::CustomCommand {
                command,
                args,
                extra_env,
                invocation_strategy,
                invocation_location,
            } => {
                let mut cmd = Command::new(command);
                cmd.envs(extra_env);

                match invocation_location {
                    InvocationLocation::Workspace => {
                        match invocation_strategy {
                            InvocationStrategy::Once => {
                                cmd.current_dir(&self.root);
                            }
                            InvocationStrategy::PerWorkspace => {
                                // FIXME: cmd.current_dir(&affected_workspace);
                                cmd.current_dir(&self.root);
                            }
                        }
                    }
                    InvocationLocation::Root(root) => {
                        cmd.current_dir(root);
                    }
                }

                if args.contains(&SAVED_FILE_PLACEHOLDER.to_owned()) {
                    // If the custom command has a $saved_file placeholder, and
                    // we're saving a file, replace the placeholder in the arguments.
                    if let Some(saved_file) = saved_file {
                        let args = args
                            .iter()
                            .map(|arg| {
                                if arg == SAVED_FILE_PLACEHOLDER {
                                    saved_file.to_string()
                                } else {
                                    arg.clone()
                                }
                            })
                            .collect();
                        (cmd, args)
                    } else {
                        // The custom command has a $saved_file placeholder,
                        // but we had an IDE event that wasn't a file save. Do nothing.
                        return None;
                    }
                } else {
                    (cmd, args.clone())
                }
            }
        };

        cmd.args(args);
        Some(cmd)
    }

    fn send(&self, check_task: Message) {
        (self.sender)(check_task);
    }
}

#[allow(clippy::large_enum_variant)]
enum CargoCheckMessage {
    CompilerArtifact(cargo_metadata::Artifact),
    Diagnostic(Diagnostic),
}

impl ParseFromLine for CargoCheckMessage {
    fn from_line(line: &str, error: &mut String) -> Option<Self> {
        let mut deserializer = serde_json::Deserializer::from_str(line);
        deserializer.disable_recursion_limit();
        if let Ok(message) = JsonMessage::deserialize(&mut deserializer) {
            return match message {
                // Skip certain kinds of messages to only spend time on what's useful
                JsonMessage::Cargo(message) => match message {
                    cargo_metadata::Message::CompilerArtifact(artifact) if !artifact.fresh => {
                        Some(CargoCheckMessage::CompilerArtifact(artifact))
                    }
                    cargo_metadata::Message::CompilerMessage(msg) => {
                        Some(CargoCheckMessage::Diagnostic(msg.message))
                    }
                    _ => None,
                },
                JsonMessage::Rustc(message) => Some(CargoCheckMessage::Diagnostic(message)),
            };
        }

        error.push_str(line);
        error.push('\n');
        None
    }

    fn from_eof() -> Option<Self> {
        None
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonMessage {
    Cargo(cargo_metadata::Message),
    Rustc(Diagnostic),
}
