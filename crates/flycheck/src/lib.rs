//! cargo_check provides the functionality needed to run `cargo check` or
//! another compatible command (f.x. clippy) in a background thread and provide
//! LSP diagnostics based on the output of the command.

use std::{
    fmt,
    io::{self, BufReader},
    ops,
    path::PathBuf,
    process::{self, Command, Stdio},
    time::Duration,
};

use crossbeam_channel::{never, select, unbounded, Receiver, Sender};

pub use cargo_metadata::diagnostic::{
    Applicability, Diagnostic, DiagnosticLevel, DiagnosticSpan, DiagnosticSpanMacroExpansion,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlycheckConfig {
    CargoCommand {
        command: String,
        all_targets: bool,
        all_features: bool,
        features: Vec<String>,
        extra_args: Vec<String>,
    },
    CustomCommand {
        command: String,
        args: Vec<String>,
    },
}

impl fmt::Display for FlycheckConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlycheckConfig::CargoCommand { command, .. } => write!(f, "cargo {}", command),
            FlycheckConfig::CustomCommand { command, args } => {
                write!(f, "{} {}", command, args.join(" "))
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
    cmd_send: Sender<Restart>,
    handle: jod_thread::JoinHandle,
}

impl FlycheckHandle {
    pub fn spawn(
        sender: Box<dyn Fn(Message) + Send>,
        config: FlycheckConfig,
        workspace_root: PathBuf,
    ) -> FlycheckHandle {
        let (cmd_send, cmd_recv) = unbounded::<Restart>();
        let handle = jod_thread::spawn(move || {
            FlycheckActor::new(sender, config, workspace_root).run(cmd_recv);
        });
        FlycheckHandle { cmd_send, handle }
    }

    /// Schedule a re-start of the cargo check worker.
    pub fn update(&self) {
        self.cmd_send.send(Restart).unwrap();
    }
}

#[derive(Debug)]
pub enum Message {
    /// Request adding a diagnostic with fixes included to a file
    AddDiagnostic { workspace_root: PathBuf, diagnostic: Diagnostic },

    /// Request check progress notification to client
    Progress(Progress),
}

#[derive(Debug)]
pub enum Progress {
    DidStart,
    DidCheckCrate(String),
    DidFinish,
    DidCancel,
}

struct Restart;

struct FlycheckActor {
    sender: Box<dyn Fn(Message) + Send>,
    config: FlycheckConfig,
    workspace_root: PathBuf,
    /// WatchThread exists to wrap around the communication needed to be able to
    /// run `cargo check` without blocking. Currently the Rust standard library
    /// doesn't provide a way to read sub-process output without blocking, so we
    /// have to wrap sub-processes output handling in a thread and pass messages
    /// back over a channel.
    // XXX: drop order is significant
    check_process: Option<(Receiver<cargo_metadata::Message>, jod_thread::JoinHandle)>,
}

enum Event {
    Restart(Restart),
    CheckEvent(Option<cargo_metadata::Message>),
}

impl FlycheckActor {
    fn new(
        sender: Box<dyn Fn(Message) + Send>,
        config: FlycheckConfig,
        workspace_root: PathBuf,
    ) -> FlycheckActor {
        FlycheckActor { sender, config, workspace_root, check_process: None }
    }
    fn next_event(&self, inbox: &Receiver<Restart>) -> Option<Event> {
        let check_chan = self.check_process.as_ref().map(|(chan, _thread)| chan);
        select! {
            recv(inbox) -> msg => msg.ok().map(Event::Restart),
            recv(check_chan.unwrap_or(&never())) -> msg => Some(Event::CheckEvent(msg.ok())),
        }
    }
    fn run(&mut self, inbox: Receiver<Restart>) {
        while let Some(event) = self.next_event(&inbox) {
            match event {
                Event::Restart(Restart) => {
                    while let Ok(Restart) = inbox.recv_timeout(Duration::from_millis(50)) {}
                    self.cancel_check_process();
                    self.check_process = Some(self.start_check_process());
                    self.send(Message::Progress(Progress::DidStart));
                }
                Event::CheckEvent(None) => {
                    // Watcher finished, replace it with a never channel to
                    // avoid busy-waiting.
                    assert!(self.check_process.take().is_some());
                    self.send(Message::Progress(Progress::DidFinish));
                }
                Event::CheckEvent(Some(message)) => match message {
                    cargo_metadata::Message::CompilerArtifact(msg) => {
                        self.send(Message::Progress(Progress::DidCheckCrate(msg.target.name)));
                    }

                    cargo_metadata::Message::CompilerMessage(msg) => {
                        self.send(Message::AddDiagnostic {
                            workspace_root: self.workspace_root.clone(),
                            diagnostic: msg.message,
                        });
                    }

                    cargo_metadata::Message::BuildScriptExecuted(_)
                    | cargo_metadata::Message::BuildFinished(_)
                    | cargo_metadata::Message::TextLine(_)
                    | cargo_metadata::Message::Unknown => {}
                },
            }
        }
        // If we rerun the thread, we need to discard the previous check results first
        self.cancel_check_process();
    }
    fn cancel_check_process(&mut self) {
        if self.check_process.take().is_some() {
            self.send(Message::Progress(Progress::DidCancel));
        }
    }
    fn start_check_process(&self) -> (Receiver<cargo_metadata::Message>, jod_thread::JoinHandle) {
        let mut cmd = match &self.config {
            FlycheckConfig::CargoCommand {
                command,
                all_targets,
                all_features,
                extra_args,
                features,
            } => {
                let mut cmd = Command::new(ra_toolchain::cargo());
                cmd.arg(command);
                cmd.args(&["--workspace", "--message-format=json", "--manifest-path"])
                    .arg(self.workspace_root.join("Cargo.toml"));
                if *all_targets {
                    cmd.arg("--all-targets");
                }
                if *all_features {
                    cmd.arg("--all-features");
                } else if !features.is_empty() {
                    cmd.arg("--features");
                    cmd.arg(features.join(" "));
                }
                cmd.args(extra_args);
                cmd
            }
            FlycheckConfig::CustomCommand { command, args } => {
                let mut cmd = Command::new(command);
                cmd.args(args);
                cmd
            }
        };
        cmd.current_dir(&self.workspace_root);

        let (message_send, message_recv) = unbounded();
        let thread = jod_thread::spawn(move || {
            // If we trigger an error here, we will do so in the loop instead,
            // which will break out of the loop, and continue the shutdown
            let res = run_cargo(cmd, &mut |message| {
                // Skip certain kinds of messages to only spend time on what's useful
                match &message {
                    cargo_metadata::Message::CompilerArtifact(artifact) if artifact.fresh => {
                        return true
                    }
                    cargo_metadata::Message::BuildScriptExecuted(_)
                    | cargo_metadata::Message::Unknown => return true,
                    _ => {}
                }

                // if the send channel was closed, we want to shutdown
                message_send.send(message).is_ok()
            });

            if let Err(err) = res {
                // FIXME: make the `message_send` to be `Sender<Result<CheckEvent, CargoError>>`
                // to display user-caused misconfiguration errors instead of just logging them here
                log::error!("Cargo watcher failed {:?}", err);
            }
        });
        (message_recv, thread)
    }

    fn send(&self, check_task: Message) {
        (self.sender)(check_task)
    }
}

fn run_cargo(
    mut command: Command,
    on_message: &mut dyn FnMut(cargo_metadata::Message) -> bool,
) -> io::Result<()> {
    let child =
        command.stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null()).spawn()?;
    let mut child = ChildKiller(child);

    // We manually read a line at a time, instead of using serde's
    // stream deserializers, because the deserializer cannot recover
    // from an error, resulting in it getting stuck, because we try to
    // be resillient against failures.
    //
    // Because cargo only outputs one JSON object per line, we can
    // simply skip a line if it doesn't parse, which just ignores any
    // erroneus output.
    let stdout = BufReader::new(child.stdout.take().unwrap());
    let mut read_at_least_one_message = false;
    for message in cargo_metadata::Message::parse_stream(stdout) {
        let message = match message {
            Ok(message) => message,
            Err(err) => {
                log::error!("Invalid json from cargo check, ignoring ({})", err);
                continue;
            }
        };

        read_at_least_one_message = true;

        if !on_message(message) {
            break;
        }
    }

    // It is okay to ignore the result, as it only errors if the process is already dead
    let _ = child.kill();

    let exit_status = child.wait()?;
    if !exit_status.success() && !read_at_least_one_message {
        // FIXME: Read the stderr to display the reason, see `read2()` reference in PR comment:
        // https://github.com/rust-analyzer/rust-analyzer/pull/3632#discussion_r395605298
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "the command produced no valid metadata (exit code: {:?}): {:?}",
                exit_status, command
            ),
        ));
    }

    Ok(())
}

struct ChildKiller(process::Child);

impl ops::Deref for ChildKiller {
    type Target = process::Child;
    fn deref(&self) -> &process::Child {
        &self.0
    }
}

impl ops::DerefMut for ChildKiller {
    fn deref_mut(&mut self) -> &mut process::Child {
        &mut self.0
    }
}

impl Drop for ChildKiller {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}
