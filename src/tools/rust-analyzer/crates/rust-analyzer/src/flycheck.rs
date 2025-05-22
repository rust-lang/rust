//! Flycheck provides the functionality needed to run `cargo check` to provide
//! LSP diagnostics based on the output of the command.

use std::{fmt, io, process::Command, time::Duration};

use cargo_metadata::PackageId;
use crossbeam_channel::{Receiver, Sender, select_biased, unbounded};
use ide_db::FxHashSet;
use paths::{AbsPath, AbsPathBuf, Utf8PathBuf};
use rustc_hash::FxHashMap;
use serde::Deserialize as _;
use serde_derive::Deserialize;

pub(crate) use cargo_metadata::diagnostic::{
    Applicability, Diagnostic, DiagnosticCode, DiagnosticLevel, DiagnosticSpan,
};
use toolchain::Tool;
use triomphe::Arc;

use crate::command::{CargoParser, CommandHandle};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum InvocationStrategy {
    Once,
    #[default]
    PerWorkspace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CargoOptions {
    pub(crate) target_tuples: Vec<String>,
    pub(crate) all_targets: bool,
    pub(crate) no_default_features: bool,
    pub(crate) all_features: bool,
    pub(crate) features: Vec<String>,
    pub(crate) extra_args: Vec<String>,
    pub(crate) extra_test_bin_args: Vec<String>,
    pub(crate) extra_env: FxHashMap<String, Option<String>>,
    pub(crate) target_dir: Option<Utf8PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) enum Target {
    Bin(String),
    Example(String),
    Benchmark(String),
    Test(String),
}

impl CargoOptions {
    pub(crate) fn apply_on_command(&self, cmd: &mut Command) {
        for target in &self.target_tuples {
            cmd.args(["--target", target.as_str()]);
        }
        if self.all_targets {
            cmd.arg("--all-targets");
        }
        if self.all_features {
            cmd.arg("--all-features");
        } else {
            if self.no_default_features {
                cmd.arg("--no-default-features");
            }
            if !self.features.is_empty() {
                cmd.arg("--features");
                cmd.arg(self.features.join(" "));
            }
        }
        if let Some(target_dir) = &self.target_dir {
            cmd.arg("--target-dir").arg(target_dir);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FlycheckConfig {
    CargoCommand {
        command: String,
        options: CargoOptions,
        ansi_color_output: bool,
    },
    CustomCommand {
        command: String,
        args: Vec<String>,
        extra_env: FxHashMap<String, Option<String>>,
        invocation_strategy: InvocationStrategy,
    },
}

impl FlycheckConfig {
    pub(crate) fn invocation_strategy_once(&self) -> bool {
        match self {
            FlycheckConfig::CargoCommand { .. } => false,
            FlycheckConfig::CustomCommand { invocation_strategy, .. } => {
                *invocation_strategy == InvocationStrategy::Once
            }
        }
    }
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
pub(crate) struct FlycheckHandle {
    // XXX: drop order is significant
    sender: Sender<StateChange>,
    _thread: stdx::thread::JoinHandle,
    id: usize,
}

impl FlycheckHandle {
    pub(crate) fn spawn(
        id: usize,
        sender: Sender<FlycheckMessage>,
        config: FlycheckConfig,
        sysroot_root: Option<AbsPathBuf>,
        workspace_root: AbsPathBuf,
        manifest_path: Option<AbsPathBuf>,
    ) -> FlycheckHandle {
        let actor =
            FlycheckActor::new(id, sender, config, sysroot_root, workspace_root, manifest_path);
        let (sender, receiver) = unbounded::<StateChange>();
        let thread =
            stdx::thread::Builder::new(stdx::thread::ThreadIntent::Worker, format!("Flycheck{id}"))
                .spawn(move || actor.run(receiver))
                .expect("failed to spawn thread");
        FlycheckHandle { id, sender, _thread: thread }
    }

    /// Schedule a re-start of the cargo check worker to do a workspace wide check.
    pub(crate) fn restart_workspace(&self, saved_file: Option<AbsPathBuf>) {
        self.sender.send(StateChange::Restart { package: None, saved_file, target: None }).unwrap();
    }

    /// Schedule a re-start of the cargo check worker to do a package wide check.
    pub(crate) fn restart_for_package(&self, package: String, target: Option<Target>) {
        self.sender
            .send(StateChange::Restart { package: Some(package), saved_file: None, target })
            .unwrap();
    }

    /// Stop this cargo check worker.
    pub(crate) fn cancel(&self) {
        self.sender.send(StateChange::Cancel).unwrap();
    }

    pub(crate) fn id(&self) -> usize {
        self.id
    }
}

pub(crate) enum FlycheckMessage {
    /// Request adding a diagnostic with fixes included to a file
    AddDiagnostic {
        id: usize,
        workspace_root: Arc<AbsPathBuf>,
        diagnostic: Diagnostic,
        package_id: Option<Arc<PackageId>>,
    },

    /// Request clearing all outdated diagnostics.
    ClearDiagnostics {
        id: usize,
        /// The package whose diagnostics to clear, or if unspecified, all diagnostics.
        package_id: Option<Arc<PackageId>>,
    },

    /// Request check progress notification to client
    Progress {
        /// Flycheck instance ID
        id: usize,
        progress: Progress,
    },
}

impl fmt::Debug for FlycheckMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlycheckMessage::AddDiagnostic { id, workspace_root, diagnostic, package_id } => f
                .debug_struct("AddDiagnostic")
                .field("id", id)
                .field("workspace_root", workspace_root)
                .field("package_id", package_id)
                .field("diagnostic_code", &diagnostic.code.as_ref().map(|it| &it.code))
                .finish(),
            FlycheckMessage::ClearDiagnostics { id, package_id } => f
                .debug_struct("ClearDiagnostics")
                .field("id", id)
                .field("package_id", package_id)
                .finish(),
            FlycheckMessage::Progress { id, progress } => {
                f.debug_struct("Progress").field("id", id).field("progress", progress).finish()
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum Progress {
    DidStart,
    DidCheckCrate(String),
    DidFinish(io::Result<()>),
    DidCancel,
    DidFailToRestart(String),
}

enum StateChange {
    Restart { package: Option<String>, saved_file: Option<AbsPathBuf>, target: Option<Target> },
    Cancel,
}

/// A [`FlycheckActor`] is a single check instance of a workspace.
struct FlycheckActor {
    /// The workspace id of this flycheck instance.
    id: usize,

    sender: Sender<FlycheckMessage>,
    config: FlycheckConfig,
    manifest_path: Option<AbsPathBuf>,
    /// Either the workspace root of the workspace we are flychecking,
    /// or the project root of the project.
    root: Arc<AbsPathBuf>,
    sysroot_root: Option<AbsPathBuf>,
    /// CargoHandle exists to wrap around the communication needed to be able to
    /// run `cargo check` without blocking. Currently the Rust standard library
    /// doesn't provide a way to read sub-process output without blocking, so we
    /// have to wrap sub-processes output handling in a thread and pass messages
    /// back over a channel.
    command_handle: Option<CommandHandle<CargoCheckMessage>>,
    /// The receiver side of the channel mentioned above.
    command_receiver: Option<Receiver<CargoCheckMessage>>,
    diagnostics_cleared_for: FxHashSet<Arc<PackageId>>,
    diagnostics_received: DiagnosticsReceived,
}

#[derive(PartialEq)]
enum DiagnosticsReceived {
    Yes,
    No,
    YesAndClearedForAll,
}

#[allow(clippy::large_enum_variant)]
enum Event {
    RequestStateChange(StateChange),
    CheckEvent(Option<CargoCheckMessage>),
}

pub(crate) const SAVED_FILE_PLACEHOLDER: &str = "$saved_file";

impl FlycheckActor {
    fn new(
        id: usize,
        sender: Sender<FlycheckMessage>,
        config: FlycheckConfig,
        sysroot_root: Option<AbsPathBuf>,
        workspace_root: AbsPathBuf,
        manifest_path: Option<AbsPathBuf>,
    ) -> FlycheckActor {
        tracing::info!(%id, ?workspace_root, "Spawning flycheck");
        FlycheckActor {
            id,
            sender,
            config,
            sysroot_root,
            root: Arc::new(workspace_root),
            manifest_path,
            command_handle: None,
            command_receiver: None,
            diagnostics_cleared_for: Default::default(),
            diagnostics_received: DiagnosticsReceived::No,
        }
    }

    fn report_progress(&self, progress: Progress) {
        self.send(FlycheckMessage::Progress { id: self.id, progress });
    }

    fn next_event(&self, inbox: &Receiver<StateChange>) -> Option<Event> {
        let Some(command_receiver) = &self.command_receiver else {
            return inbox.recv().ok().map(Event::RequestStateChange);
        };

        // Biased to give restarts a preference so check outputs don't block a restart or stop
        select_biased! {
            recv(inbox) -> msg => msg.ok().map(Event::RequestStateChange),
            recv(command_receiver) -> msg => Some(Event::CheckEvent(msg.ok())),
        }
    }

    fn run(mut self, inbox: Receiver<StateChange>) {
        'event: while let Some(event) = self.next_event(&inbox) {
            match event {
                Event::RequestStateChange(StateChange::Cancel) => {
                    tracing::debug!(flycheck_id = self.id, "flycheck cancelled");
                    self.cancel_check_process();
                }
                Event::RequestStateChange(StateChange::Restart { package, saved_file, target }) => {
                    // Cancel the previously spawned process
                    self.cancel_check_process();
                    while let Ok(restart) = inbox.recv_timeout(Duration::from_millis(50)) {
                        // restart chained with a stop, so just cancel
                        if let StateChange::Cancel = restart {
                            continue 'event;
                        }
                    }

                    let Some(command) =
                        self.check_command(package.as_deref(), saved_file.as_deref(), target)
                    else {
                        continue;
                    };

                    let formatted_command = format!("{command:?}");

                    tracing::debug!(?command, "will restart flycheck");
                    let (sender, receiver) = unbounded();
                    match CommandHandle::spawn(command, CargoCheckParser, sender) {
                        Ok(command_handle) => {
                            tracing::debug!(command = formatted_command, "did restart flycheck");
                            self.command_handle = Some(command_handle);
                            self.command_receiver = Some(receiver);
                            self.report_progress(Progress::DidStart);
                        }
                        Err(error) => {
                            self.report_progress(Progress::DidFailToRestart(format!(
                                "Failed to run the following command: {formatted_command} error={error}"
                            )));
                        }
                    }
                }
                Event::CheckEvent(None) => {
                    tracing::debug!(flycheck_id = self.id, "flycheck finished");

                    // Watcher finished
                    let command_handle = self.command_handle.take().unwrap();
                    self.command_receiver.take();
                    let formatted_handle = format!("{command_handle:?}");

                    let res = command_handle.join();
                    if let Err(error) = &res {
                        tracing::error!(
                            "Flycheck failed to run the following command: {}, error={}",
                            formatted_handle,
                            error
                        );
                    }
                    if self.diagnostics_received == DiagnosticsReceived::No {
                        tracing::trace!(flycheck_id = self.id, "clearing diagnostics");
                        // We finished without receiving any diagnostics.
                        // Clear everything for good measure
                        self.send(FlycheckMessage::ClearDiagnostics {
                            id: self.id,
                            package_id: None,
                        });
                    }
                    self.clear_diagnostics_state();

                    self.report_progress(Progress::DidFinish(res));
                }
                Event::CheckEvent(Some(message)) => match message {
                    CargoCheckMessage::CompilerArtifact(msg) => {
                        tracing::trace!(
                            flycheck_id = self.id,
                            artifact = msg.target.name,
                            package_id = msg.package_id.repr,
                            "artifact received"
                        );
                        self.report_progress(Progress::DidCheckCrate(msg.target.name));
                        let package_id = Arc::new(msg.package_id);
                        if self.diagnostics_cleared_for.insert(package_id.clone()) {
                            tracing::trace!(
                                flycheck_id = self.id,
                                package_id = package_id.repr,
                                "clearing diagnostics"
                            );
                            self.send(FlycheckMessage::ClearDiagnostics {
                                id: self.id,
                                package_id: Some(package_id),
                            });
                        }
                    }
                    CargoCheckMessage::Diagnostic { diagnostic, package_id } => {
                        tracing::trace!(
                            flycheck_id = self.id,
                            message = diagnostic.message,
                            package_id = package_id.as_ref().map(|it| &it.repr),
                            "diagnostic received"
                        );
                        if self.diagnostics_received == DiagnosticsReceived::No {
                            self.diagnostics_received = DiagnosticsReceived::Yes;
                        }
                        if let Some(package_id) = &package_id {
                            if self.diagnostics_cleared_for.insert(package_id.clone()) {
                                tracing::trace!(
                                    flycheck_id = self.id,
                                    package_id = package_id.repr,
                                    "clearing diagnostics"
                                );
                                self.send(FlycheckMessage::ClearDiagnostics {
                                    id: self.id,
                                    package_id: Some(package_id.clone()),
                                });
                            }
                        } else if self.diagnostics_received
                            != DiagnosticsReceived::YesAndClearedForAll
                        {
                            self.diagnostics_received = DiagnosticsReceived::YesAndClearedForAll;
                            self.send(FlycheckMessage::ClearDiagnostics {
                                id: self.id,
                                package_id: None,
                            });
                        }
                        self.send(FlycheckMessage::AddDiagnostic {
                            id: self.id,
                            package_id,
                            workspace_root: self.root.clone(),
                            diagnostic,
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
            self.command_receiver.take();
            self.report_progress(Progress::DidCancel);
        }
        self.clear_diagnostics_state();
    }

    fn clear_diagnostics_state(&mut self) {
        self.diagnostics_cleared_for.clear();
        self.diagnostics_received = DiagnosticsReceived::No;
    }

    /// Construct a `Command` object for checking the user's code. If the user
    /// has specified a custom command with placeholders that we cannot fill,
    /// return None.
    fn check_command(
        &self,
        package: Option<&str>,
        saved_file: Option<&AbsPath>,
        target: Option<Target>,
    ) -> Option<Command> {
        match &self.config {
            FlycheckConfig::CargoCommand { command, options, ansi_color_output } => {
                let mut cmd =
                    toolchain::command(Tool::Cargo.path(), &*self.root, &options.extra_env);
                if let Some(sysroot_root) = &self.sysroot_root {
                    if !options.extra_env.contains_key("RUSTUP_TOOLCHAIN")
                        && std::env::var_os("RUSTUP_TOOLCHAIN").is_none()
                    {
                        cmd.env("RUSTUP_TOOLCHAIN", AsRef::<std::path::Path>::as_ref(sysroot_root));
                    }
                }
                cmd.arg(command);

                match package {
                    Some(pkg) => cmd.arg("-p").arg(pkg),
                    None => cmd.arg("--workspace"),
                };

                if let Some(tgt) = target {
                    match tgt {
                        Target::Bin(tgt) => cmd.arg("--bin").arg(tgt),
                        Target::Example(tgt) => cmd.arg("--example").arg(tgt),
                        Target::Test(tgt) => cmd.arg("--test").arg(tgt),
                        Target::Benchmark(tgt) => cmd.arg("--bench").arg(tgt),
                    };
                }

                cmd.arg(if *ansi_color_output {
                    "--message-format=json-diagnostic-rendered-ansi"
                } else {
                    "--message-format=json"
                });

                if let Some(manifest_path) = &self.manifest_path {
                    cmd.arg("--manifest-path");
                    cmd.arg(manifest_path);
                    if manifest_path.extension() == Some("rs") {
                        cmd.arg("-Zscript");
                    }
                }

                cmd.arg("--keep-going");

                options.apply_on_command(&mut cmd);
                cmd.args(&options.extra_args);
                Some(cmd)
            }
            FlycheckConfig::CustomCommand { command, args, extra_env, invocation_strategy } => {
                let root = match invocation_strategy {
                    InvocationStrategy::Once => &*self.root,
                    InvocationStrategy::PerWorkspace => {
                        // FIXME: &affected_workspace
                        &*self.root
                    }
                };
                let mut cmd = toolchain::command(command, root, extra_env);

                // If the custom command has a $saved_file placeholder, and
                // we're saving a file, replace the placeholder in the arguments.
                if let Some(saved_file) = saved_file {
                    for arg in args {
                        if arg == SAVED_FILE_PLACEHOLDER {
                            cmd.arg(saved_file);
                        } else {
                            cmd.arg(arg);
                        }
                    }
                } else {
                    for arg in args {
                        if arg == SAVED_FILE_PLACEHOLDER {
                            // The custom command has a $saved_file placeholder,
                            // but we had an IDE event that wasn't a file save. Do nothing.
                            return None;
                        }

                        cmd.arg(arg);
                    }
                }

                Some(cmd)
            }
        }
    }

    #[track_caller]
    fn send(&self, check_task: FlycheckMessage) {
        self.sender.send(check_task).unwrap();
    }
}

#[allow(clippy::large_enum_variant)]
enum CargoCheckMessage {
    CompilerArtifact(cargo_metadata::Artifact),
    Diagnostic { diagnostic: Diagnostic, package_id: Option<Arc<PackageId>> },
}

struct CargoCheckParser;

impl CargoParser<CargoCheckMessage> for CargoCheckParser {
    fn from_line(&self, line: &str, error: &mut String) -> Option<CargoCheckMessage> {
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
                        Some(CargoCheckMessage::Diagnostic {
                            diagnostic: msg.message,
                            package_id: Some(Arc::new(msg.package_id)),
                        })
                    }
                    _ => None,
                },
                JsonMessage::Rustc(message) => {
                    Some(CargoCheckMessage::Diagnostic { diagnostic: message, package_id: None })
                }
            };
        }

        error.push_str(line);
        error.push('\n');
        None
    }

    fn from_eof(&self) -> Option<CargoCheckMessage> {
        None
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonMessage {
    Cargo(cargo_metadata::Message),
    Rustc(Diagnostic),
}
