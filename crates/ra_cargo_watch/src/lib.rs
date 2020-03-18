//! cargo_check provides the functionality needed to run `cargo check` or
//! another compatible command (f.x. clippy) in a background thread and provide
//! LSP diagnostics based on the output of the command.
use cargo_metadata::Message;
use crossbeam_channel::{never, select, unbounded, Receiver, RecvError, Sender};
use lsp_types::{
    CodeAction, CodeActionOrCommand, Diagnostic, Url, WorkDoneProgress, WorkDoneProgressBegin,
    WorkDoneProgressEnd, WorkDoneProgressReport,
};
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread::JoinHandle,
    time::Instant,
};

mod conv;

use crate::conv::{map_rust_diagnostic_to_lsp, MappedRustDiagnostic};

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
    cmd_send: Option<Sender<CheckCommand>>,
    handle: Option<JoinHandle<()>>,
}

impl CheckWatcher {
    pub fn new(options: &CheckOptions, workspace_root: PathBuf) -> CheckWatcher {
        let options = options.clone();

        let (task_send, task_recv) = unbounded::<CheckTask>();
        let (cmd_send, cmd_recv) = unbounded::<CheckCommand>();
        let handle = std::thread::spawn(move || {
            let mut check = CheckWatcherThread::new(options, workspace_root);
            check.run(&task_send, &cmd_recv);
        });
        CheckWatcher { task_recv, cmd_send: Some(cmd_send), handle: Some(handle) }
    }

    /// Returns a CheckWatcher that doesn't actually do anything
    pub fn dummy() -> CheckWatcher {
        CheckWatcher { task_recv: never(), cmd_send: None, handle: None }
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
pub enum CheckTask {
    /// Request a clearing of all cached diagnostics from the check watcher
    ClearDiagnostics,

    /// Request adding a diagnostic with fixes included to a file
    AddDiagnostic { url: Url, diagnostic: Diagnostic, fixes: Vec<CodeActionOrCommand> },

    /// Request check progress notification to client
    Status(WorkDoneProgress),
}

pub enum CheckCommand {
    /// Request re-start of check thread
    Update,
}

struct CheckWatcherThread {
    options: CheckOptions,
    workspace_root: PathBuf,
    watcher: WatchThread,
    last_update_req: Option<Instant>,
}

impl CheckWatcherThread {
    fn new(options: CheckOptions, workspace_root: PathBuf) -> CheckWatcherThread {
        CheckWatcherThread {
            options,
            workspace_root,
            watcher: WatchThread::dummy(),
            last_update_req: None,
        }
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
                task_send.send(CheckTask::ClearDiagnostics).unwrap();

                // Replace with a dummy watcher first so we drop the original and wait for completion
                std::mem::replace(&mut self.watcher, WatchThread::dummy());

                // Then create the actual new watcher
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

    fn handle_message(&self, msg: CheckEvent, task_send: &Sender<CheckTask>) {
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
                let map_result = map_rust_diagnostic_to_lsp(&msg.message, &self.workspace_root);
                if map_result.is_empty() {
                    return;
                }

                for MappedRustDiagnostic { location, diagnostic, fixes } in map_result {
                    let fixes = fixes
                        .into_iter()
                        .map(|fix| {
                            CodeAction { diagnostics: Some(vec![diagnostic.clone()]), ..fix }.into()
                        })
                        .collect();

                    task_send
                        .send(CheckTask::AddDiagnostic { url: location.uri, diagnostic, fixes })
                        .unwrap();
                }
            }

            CheckEvent::Msg(Message::BuildScriptExecuted(_msg)) => {}
            CheckEvent::Msg(Message::Unknown) => {}
        }
    }
}

#[derive(Debug)]
pub struct DiagnosticWithFixes {
    diagnostic: Diagnostic,
    fixes: Vec<CodeAction>,
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

pub fn run_cargo(
    args: &[String],
    current_dir: Option<&Path>,
    on_message: &mut dyn FnMut(cargo_metadata::Message) -> bool,
) -> Child {
    let mut command = Command::new("cargo");
    if let Some(current_dir) = current_dir {
        command.current_dir(current_dir);
    }

    let mut child = command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .expect("couldn't launch cargo");

    // We manually read a line at a time, instead of using serde's
    // stream deserializers, because the deserializer cannot recover
    // from an error, resulting in it getting stuck, because we try to
    // be resillient against failures.
    //
    // Because cargo only outputs one JSON object per line, we can
    // simply skip a line if it doesn't parse, which just ignores any
    // erroneus output.
    let stdout = BufReader::new(child.stdout.take().unwrap());
    for line in stdout.lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                log::error!("Couldn't read line from cargo: {}", err);
                continue;
            }
        };

        let message = serde_json::from_str::<cargo_metadata::Message>(&line);
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                log::error!("Invalid json from cargo check, ignoring ({}): {:?} ", err, line);
                continue;
            }
        };

        if !on_message(message) {
            break;
        }
    }

    child
}

impl WatchThread {
    fn dummy() -> WatchThread {
        WatchThread { handle: None, message_recv: never() }
    }

    fn new(options: &CheckOptions, workspace_root: &Path) -> WatchThread {
        let mut args: Vec<String> = vec![
            options.command.clone(),
            "--workspace".to_string(),
            "--message-format=json".to_string(),
            "--manifest-path".to_string(),
            format!("{}/Cargo.toml", workspace_root.display()),
        ];
        if options.all_targets {
            args.push("--all-targets".to_string());
        }
        args.extend(options.args.iter().cloned());

        let (message_send, message_recv) = unbounded();
        let workspace_root = workspace_root.to_owned();
        let handle = if options.enable {
            Some(std::thread::spawn(move || {
                // If we trigger an error here, we will do so in the loop instead,
                // which will break out of the loop, and continue the shutdown
                let _ = message_send.send(CheckEvent::Begin);

                let mut child = run_cargo(&args, Some(&workspace_root), &mut |message| {
                    // Skip certain kinds of messages to only spend time on what's useful
                    match &message {
                        Message::CompilerArtifact(artifact) if artifact.fresh => return true,
                        Message::BuildScriptExecuted(_) => return true,
                        Message::Unknown => return true,
                        _ => {}
                    }

                    match message_send.send(CheckEvent::Msg(message)) {
                        Ok(()) => {}
                        Err(_err) => {
                            // The send channel was closed, so we want to shutdown
                            return false;
                        }
                    };

                    true
                });

                // We can ignore any error here, as we are already in the progress
                // of shutting down.
                let _ = message_send.send(CheckEvent::End);

                // It is okay to ignore the result, as it only errors if the process is already dead
                let _ = child.kill();

                // Again, we are resilient to errors, so we don't try to panic here
                match child.wait_with_output() {
                    Ok(output) => match output.status.code() {
                        Some(0) | None => {}
                        Some(exit_code) => {
                            let output =
                                std::str::from_utf8(&output.stderr).unwrap_or("<bad utf8 output>");

                            if !output.contains("could not compile") {
                                log::error!("Cargo failed with exit code {} {}", exit_code, output);
                            }
                        }
                    },
                    Err(err) => log::error!("Cargo io error: {:?}", err),
                }
            }))
        } else {
            None
        };
        WatchThread { handle, message_recv }
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
