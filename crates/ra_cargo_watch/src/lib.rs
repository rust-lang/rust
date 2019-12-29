//! cargo_check provides the functionality needed to run `cargo check` or
//! another compatible command (f.x. clippy) in a background thread and provide
//! LSP diagnostics based on the output of the command.
use cargo_metadata::Message;
use crossbeam_channel::{never, select, unbounded, Receiver, RecvError, Sender};
use lsp_types::{
    Diagnostic, Url, WorkDoneProgress, WorkDoneProgressBegin, WorkDoneProgressEnd,
    WorkDoneProgressReport,
};
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
    thread::JoinHandle,
    time::Instant,
};

mod conv;

use crate::conv::{map_rust_diagnostic_to_lsp, MappedRustDiagnostic, SuggestedFix};

pub use crate::conv::url_from_path_with_drive_lowercasing;

#[derive(Clone, Debug)]
pub struct CheckOptions {
    pub enable: bool,
    pub args: Vec<String>,
    pub command: String,
    pub all_targets: bool,
}

/// CheckWatcher wraps the shared state and communication machinery used for
/// running `cargo check` (or other compatible command) and providing
/// diagnostics based on the output.
/// The spawned thread is shut down when this struct is dropped.
#[derive(Debug)]
pub struct CheckWatcher {
    pub task_recv: Receiver<CheckTask>,
    pub shared: Arc<RwLock<CheckWatcherSharedState>>,
    cmd_send: Option<Sender<CheckCommand>>,
    handle: Option<JoinHandle<()>>,
}

impl CheckWatcher {
    pub fn new(options: &CheckOptions, workspace_root: PathBuf) -> CheckWatcher {
        let options = options.clone();
        let shared = Arc::new(RwLock::new(CheckWatcherSharedState::new()));

        let (task_send, task_recv) = unbounded::<CheckTask>();
        let (cmd_send, cmd_recv) = unbounded::<CheckCommand>();
        let shared_ = shared.clone();
        let handle = std::thread::spawn(move || {
            let mut check = CheckWatcherState::new(options, workspace_root, shared_);
            check.run(&task_send, &cmd_recv);
        });
        CheckWatcher { task_recv, cmd_send: Some(cmd_send), handle: Some(handle), shared }
    }

    /// Schedule a re-start of the cargo check worker.
    pub fn update(&self) {
        if let Some(cmd_send) = &self.cmd_send {
            cmd_send.send(CheckCommand::Update).unwrap();
        }
    }
}

impl std::ops::Drop for CheckWatcher {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Take the sender out of the option
            let recv = self.cmd_send.take();

            // Dropping the sender finishes the thread loop
            drop(recv);

            // Join the thread, it should finish shortly. We don't really care
            // whether it panicked, so it is safe to ignore the result
            let _ = handle.join();
        }
    }
}

#[derive(Debug)]
pub struct CheckWatcherSharedState {
    diagnostic_collection: HashMap<Url, Vec<Diagnostic>>,
    suggested_fix_collection: HashMap<Url, Vec<SuggestedFix>>,
}

impl CheckWatcherSharedState {
    fn new() -> CheckWatcherSharedState {
        CheckWatcherSharedState {
            diagnostic_collection: HashMap::new(),
            suggested_fix_collection: HashMap::new(),
        }
    }

    /// Clear the cached diagnostics, and schedule updating diagnostics by the
    /// server, to clear stale results.
    pub fn clear(&mut self, task_send: &Sender<CheckTask>) {
        let cleared_files: Vec<Url> = self.diagnostic_collection.keys().cloned().collect();

        self.diagnostic_collection.clear();
        self.suggested_fix_collection.clear();

        for uri in cleared_files {
            task_send.send(CheckTask::Update(uri.clone())).unwrap();
        }
    }

    pub fn diagnostics_for(&self, uri: &Url) -> Option<&[Diagnostic]> {
        self.diagnostic_collection.get(uri).map(|d| d.as_slice())
    }

    pub fn fixes_for(&self, uri: &Url) -> Option<&[SuggestedFix]> {
        self.suggested_fix_collection.get(uri).map(|d| d.as_slice())
    }

    fn add_diagnostic(&mut self, file_uri: Url, diagnostic: Diagnostic) {
        let diagnostics = self.diagnostic_collection.entry(file_uri).or_default();

        // If we're building multiple targets it's possible we've already seen this diagnostic
        let is_duplicate = diagnostics.iter().any(|d| are_diagnostics_equal(d, &diagnostic));
        if is_duplicate {
            return;
        }

        diagnostics.push(diagnostic);
    }

    fn add_suggested_fix_for_diagnostic(
        &mut self,
        mut suggested_fix: SuggestedFix,
        diagnostic: &Diagnostic,
    ) {
        let file_uri = suggested_fix.location.uri.clone();
        let file_suggestions = self.suggested_fix_collection.entry(file_uri).or_default();

        let existing_suggestion: Option<&mut SuggestedFix> =
            file_suggestions.iter_mut().find(|s| s == &&suggested_fix);
        if let Some(existing_suggestion) = existing_suggestion {
            // The existing suggestion also applies to this new diagnostic
            existing_suggestion.diagnostics.push(diagnostic.clone());
        } else {
            // We haven't seen this suggestion before
            suggested_fix.diagnostics.push(diagnostic.clone());
            file_suggestions.push(suggested_fix);
        }
    }
}

#[derive(Debug)]
pub enum CheckTask {
    /// Request a update of the given files diagnostics
    Update(Url),

    /// Request check progress notification to client
    Status(WorkDoneProgress),
}

pub enum CheckCommand {
    /// Request re-start of check thread
    Update,
}

struct CheckWatcherState {
    options: CheckOptions,
    workspace_root: PathBuf,
    watcher: WatchThread,
    last_update_req: Option<Instant>,
    shared: Arc<RwLock<CheckWatcherSharedState>>,
}

impl CheckWatcherState {
    fn new(
        options: CheckOptions,
        workspace_root: PathBuf,
        shared: Arc<RwLock<CheckWatcherSharedState>>,
    ) -> CheckWatcherState {
        let watcher = WatchThread::new(&options, &workspace_root);
        CheckWatcherState { options, workspace_root, watcher, last_update_req: None, shared }
    }

    fn run(&mut self, task_send: &Sender<CheckTask>, cmd_recv: &Receiver<CheckCommand>) {
        loop {
            select! {
                recv(&cmd_recv) -> cmd => match cmd {
                    Ok(cmd) => self.handle_command(cmd),
                    Err(RecvError) => {
                        // Command channel has closed, so shut down
                        break;
                    },
                },
                recv(self.watcher.message_recv) -> msg => match msg {
                    Ok(msg) => self.handle_message(msg, task_send),
                    Err(RecvError) => {
                        // Watcher finished, replace it with a never channel to
                        // avoid busy-waiting.
                        std::mem::replace(&mut self.watcher.message_recv, never());
                    },
                }
            };

            if self.should_recheck() {
                self.last_update_req.take();
                self.shared.write().clear(task_send);

                // By replacing the watcher, we drop the previous one which
                // causes it to shut down automatically.
                self.watcher = WatchThread::new(&self.options, &self.workspace_root);
            }
        }
    }

    fn should_recheck(&mut self) -> bool {
        if let Some(_last_update_req) = &self.last_update_req {
            // We currently only request an update on save, as we need up to
            // date source on disk for cargo check to do it's magic, so we
            // don't really need to debounce the requests at this point.
            return true;
        }
        false
    }

    fn handle_command(&mut self, cmd: CheckCommand) {
        match cmd {
            CheckCommand::Update => self.last_update_req = Some(Instant::now()),
        }
    }

    fn handle_message(&mut self, msg: CheckEvent, task_send: &Sender<CheckTask>) {
        match msg {
            CheckEvent::Begin => {
                task_send
                    .send(CheckTask::Status(WorkDoneProgress::Begin(WorkDoneProgressBegin {
                        title: "Running 'cargo check'".to_string(),
                        cancellable: Some(false),
                        message: None,
                        percentage: None,
                    })))
                    .unwrap();
            }

            CheckEvent::End => {
                task_send
                    .send(CheckTask::Status(WorkDoneProgress::End(WorkDoneProgressEnd {
                        message: None,
                    })))
                    .unwrap();
            }

            CheckEvent::Msg(Message::CompilerArtifact(msg)) => {
                task_send
                    .send(CheckTask::Status(WorkDoneProgress::Report(WorkDoneProgressReport {
                        cancellable: Some(false),
                        message: Some(msg.target.name),
                        percentage: None,
                    })))
                    .unwrap();
            }

            CheckEvent::Msg(Message::CompilerMessage(msg)) => {
                let map_result =
                    match map_rust_diagnostic_to_lsp(&msg.message, &self.workspace_root) {
                        Some(map_result) => map_result,
                        None => return,
                    };

                let MappedRustDiagnostic { location, diagnostic, suggested_fixes } = map_result;
                let file_uri = location.uri.clone();

                if !suggested_fixes.is_empty() {
                    for suggested_fix in suggested_fixes {
                        self.shared
                            .write()
                            .add_suggested_fix_for_diagnostic(suggested_fix, &diagnostic);
                    }
                }
                self.shared.write().add_diagnostic(file_uri, diagnostic);

                task_send.send(CheckTask::Update(location.uri)).unwrap();
            }

            CheckEvent::Msg(Message::BuildScriptExecuted(_msg)) => {}
            CheckEvent::Msg(Message::Unknown) => {}
        }
    }
}

/// WatchThread exists to wrap around the communication needed to be able to
/// run `cargo check` without blocking. Currently the Rust standard library
/// doesn't provide a way to read sub-process output without blocking, so we
/// have to wrap sub-processes output handling in a thread and pass messages
/// back over a channel.
/// The correct way to dispose of the thread is to drop it, on which the
/// sub-process will be killed, and the thread will be joined.
struct WatchThread {
    handle: Option<JoinHandle<()>>,
    message_recv: Receiver<CheckEvent>,
}

enum CheckEvent {
    Begin,
    Msg(cargo_metadata::Message),
    End,
}

impl WatchThread {
    fn new(options: &CheckOptions, workspace_root: &PathBuf) -> WatchThread {
        let mut args: Vec<String> = vec![
            options.command.clone(),
            "--message-format=json".to_string(),
            "--manifest-path".to_string(),
            format!("{}/Cargo.toml", workspace_root.to_string_lossy()),
        ];
        if options.all_targets {
            args.push("--all-targets".to_string());
        }
        args.extend(options.args.iter().cloned());

        let (message_send, message_recv) = unbounded();
        let enabled = options.enable;
        let handle = std::thread::spawn(move || {
            if !enabled {
                return;
            }

            let mut command = Command::new("cargo")
                .args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("couldn't launch cargo");

            // If we trigger an error here, we will do so in the loop instead,
            // which will break out of the loop, and continue the shutdown
            let _ = message_send.send(CheckEvent::Begin);

            for message in cargo_metadata::parse_messages(command.stdout.take().unwrap()) {
                let message = match message {
                    Ok(message) => message,
                    Err(err) => {
                        log::error!("Invalid json from cargo check, ignoring: {}", err);
                        continue;
                    }
                };

                match message_send.send(CheckEvent::Msg(message)) {
                    Ok(()) => {}
                    Err(_err) => {
                        // The send channel was closed, so we want to shutdown
                        break;
                    }
                }
            }

            // We can ignore any error here, as we are already in the progress
            // of shutting down.
            let _ = message_send.send(CheckEvent::End);

            // It is okay to ignore the result, as it only errors if the process is already dead
            let _ = command.kill();

            // Again, we don't care about the exit status so just ignore the result
            let _ = command.wait();
        });
        WatchThread { handle: Some(handle), message_recv }
    }
}

impl std::ops::Drop for WatchThread {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Replace our reciever with dummy one, so we can drop and close the
            // one actually communicating with the thread
            let recv = std::mem::replace(&mut self.message_recv, never());

            // Dropping the original reciever initiates thread sub-process shutdown
            drop(recv);

            // Join the thread, it should finish shortly. We don't really care
            // whether it panicked, so it is safe to ignore the result
            let _ = handle.join();
        }
    }
}

fn are_diagnostics_equal(left: &Diagnostic, right: &Diagnostic) -> bool {
    left.source == right.source
        && left.severity == right.severity
        && left.range == right.range
        && left.message == right.message
}
