use cargo_metadata::{
    diagnostic::{
        Applicability, Diagnostic as RustDiagnostic, DiagnosticLevel, DiagnosticSpan,
        DiagnosticSpanMacroExpansion,
    },
    Message,
};
use crossbeam_channel::{select, unbounded, Receiver, RecvError, Sender, TryRecvError};
use lsp_types::{
    Diagnostic, DiagnosticRelatedInformation, DiagnosticSeverity, DiagnosticTag, Location,
    NumberOrString, Position, Range, Url,
};
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    fmt::Write,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
    thread::JoinHandle,
    time::Instant,
};

#[derive(Debug)]
pub struct CheckWatcher {
    pub task_recv: Receiver<CheckTask>,
    pub cmd_send: Sender<CheckCommand>,
    pub shared: Arc<RwLock<CheckWatcherSharedState>>,
    handle: JoinHandle<()>,
}

impl CheckWatcher {
    pub fn new(workspace_root: PathBuf) -> CheckWatcher {
        let shared = Arc::new(RwLock::new(CheckWatcherSharedState::new()));

        let (task_send, task_recv) = unbounded::<CheckTask>();
        let (cmd_send, cmd_recv) = unbounded::<CheckCommand>();
        let shared_ = shared.clone();
        let handle = std::thread::spawn(move || {
            let mut check = CheckWatcherState::new(shared_, workspace_root);
            check.run(&task_send, &cmd_recv);
        });

        CheckWatcher { task_recv, cmd_send, handle, shared }
    }

    pub fn update(&self) {
        self.cmd_send.send(CheckCommand::Update).unwrap();
    }
}

pub struct CheckWatcherState {
    workspace_root: PathBuf,
    running: bool,
    watcher: WatchThread,
    last_update_req: Option<Instant>,
    shared: Arc<RwLock<CheckWatcherSharedState>>,
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
    Update(Url),
}

pub enum CheckCommand {
    Update,
}

impl CheckWatcherState {
    pub fn new(
        shared: Arc<RwLock<CheckWatcherSharedState>>,
        workspace_root: PathBuf,
    ) -> CheckWatcherState {
        let watcher = WatchThread::new(&workspace_root);
        CheckWatcherState { workspace_root, running: false, watcher, last_update_req: None, shared }
    }

    pub fn run(&mut self, task_send: &Sender<CheckTask>, cmd_recv: &Receiver<CheckCommand>) {
        self.running = true;
        while self.running {
            select! {
                recv(&cmd_recv) -> cmd => match cmd {
                    Ok(cmd) => self.handle_command(cmd),
                    Err(RecvError) => {
                        // Command channel has closed, so shut down
                        self.running = false;
                    },
                },
                recv(self.watcher.message_recv) -> msg => match msg {
                    Ok(msg) => self.handle_message(msg, task_send),
                    Err(RecvError) => {},
                }
            };

            if self.should_recheck() {
                self.last_update_req.take();
                self.shared.write().clear(task_send);

                self.watcher.cancel();
                self.watcher = WatchThread::new(&self.workspace_root);
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

    fn handle_message(&mut self, msg: cargo_metadata::Message, task_send: &Sender<CheckTask>) {
        match msg {
            Message::CompilerArtifact(_msg) => {
                // TODO: Status display
            }

            Message::CompilerMessage(msg) => {
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

            Message::BuildScriptExecuted(_msg) => {}
            Message::Unknown => {}
        }
    }
}

/// WatchThread exists to wrap around the communication needed to be able to
/// run `cargo check` without blocking. Currently the Rust standard library
/// doesn't provide a way to read sub-process output without blocking, so we
/// have to wrap sub-processes output handling in a thread and pass messages
/// back over a channel.
struct WatchThread {
    message_recv: Receiver<cargo_metadata::Message>,
    cancel_send: Sender<()>,
}

impl WatchThread {
    fn new(workspace_root: &PathBuf) -> WatchThread {
        let manifest_path = format!("{}/Cargo.toml", workspace_root.to_string_lossy());
        let (message_send, message_recv) = unbounded();
        let (cancel_send, cancel_recv) = unbounded();
        std::thread::spawn(move || {
            let mut command = Command::new("cargo")
                .args(&["check", "--message-format=json", "--manifest-path", &manifest_path])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("couldn't launch cargo");

            for message in cargo_metadata::parse_messages(command.stdout.take().unwrap()) {
                match cancel_recv.try_recv() {
                    Ok(()) | Err(TryRecvError::Disconnected) => {
                        command.kill().expect("couldn't kill command");
                    }
                    Err(TryRecvError::Empty) => (),
                }

                message_send.send(message.unwrap()).unwrap();
            }
        });
        WatchThread { message_recv, cancel_send }
    }

    fn cancel(&self) {
        let _ = self.cancel_send.send(());
    }
}

/// Converts a Rust level string to a LSP severity
fn map_level_to_severity(val: DiagnosticLevel) -> Option<DiagnosticSeverity> {
    match val {
        DiagnosticLevel::Ice => Some(DiagnosticSeverity::Error),
        DiagnosticLevel::Error => Some(DiagnosticSeverity::Error),
        DiagnosticLevel::Warning => Some(DiagnosticSeverity::Warning),
        DiagnosticLevel::Note => Some(DiagnosticSeverity::Information),
        DiagnosticLevel::Help => Some(DiagnosticSeverity::Hint),
        DiagnosticLevel::Unknown => None,
    }
}

/// Check whether a file name is from macro invocation
fn is_from_macro(file_name: &str) -> bool {
    file_name.starts_with('<') && file_name.ends_with('>')
}

/// Converts a Rust macro span to a LSP location recursively
fn map_macro_span_to_location(
    span_macro: &DiagnosticSpanMacroExpansion,
    workspace_root: &PathBuf,
) -> Option<Location> {
    if !is_from_macro(&span_macro.span.file_name) {
        return Some(map_span_to_location(&span_macro.span, workspace_root));
    }

    if let Some(expansion) = &span_macro.span.expansion {
        return map_macro_span_to_location(&expansion, workspace_root);
    }

    None
}

/// Converts a Rust span to a LSP location
fn map_span_to_location(span: &DiagnosticSpan, workspace_root: &PathBuf) -> Location {
    if is_from_macro(&span.file_name) && span.expansion.is_some() {
        let expansion = span.expansion.as_ref().unwrap();
        if let Some(macro_range) = map_macro_span_to_location(&expansion, workspace_root) {
            return macro_range;
        }
    }

    let mut file_name = workspace_root.clone();
    file_name.push(&span.file_name);
    let uri = Url::from_file_path(file_name).unwrap();

    let range = Range::new(
        Position::new(span.line_start as u64 - 1, span.column_start as u64 - 1),
        Position::new(span.line_end as u64 - 1, span.column_end as u64 - 1),
    );

    Location { uri, range }
}

/// Converts a secondary Rust span to a LSP related information
///
/// If the span is unlabelled this will return `None`.
fn map_secondary_span_to_related(
    span: &DiagnosticSpan,
    workspace_root: &PathBuf,
) -> Option<DiagnosticRelatedInformation> {
    if let Some(label) = &span.label {
        let location = map_span_to_location(span, workspace_root);
        Some(DiagnosticRelatedInformation { location, message: label.clone() })
    } else {
        // Nothing to label this with
        None
    }
}

/// Determines if diagnostic is related to unused code
fn is_unused_or_unnecessary(rd: &RustDiagnostic) -> bool {
    if let Some(code) = &rd.code {
        match code.code.as_str() {
            "dead_code" | "unknown_lints" | "unreachable_code" | "unused_attributes"
            | "unused_imports" | "unused_macros" | "unused_variables" => true,
            _ => false,
        }
    } else {
        false
    }
}

/// Determines if diagnostic is related to deprecated code
fn is_deprecated(rd: &RustDiagnostic) -> bool {
    if let Some(code) = &rd.code {
        match code.code.as_str() {
            "deprecated" => true,
            _ => false,
        }
    } else {
        false
    }
}

#[derive(Debug)]
pub struct SuggestedFix {
    pub title: String,
    pub location: Location,
    pub replacement: String,
    pub applicability: Applicability,
    pub diagnostics: Vec<Diagnostic>,
}

impl std::cmp::PartialEq<SuggestedFix> for SuggestedFix {
    fn eq(&self, other: &SuggestedFix) -> bool {
        if self.title == other.title
            && self.location == other.location
            && self.replacement == other.replacement
        {
            // Applicability doesn't impl PartialEq...
            match (&self.applicability, &other.applicability) {
                (Applicability::MachineApplicable, Applicability::MachineApplicable) => true,
                (Applicability::HasPlaceholders, Applicability::HasPlaceholders) => true,
                (Applicability::MaybeIncorrect, Applicability::MaybeIncorrect) => true,
                (Applicability::Unspecified, Applicability::Unspecified) => true,
                _ => false,
            }
        } else {
            false
        }
    }
}

enum MappedRustChildDiagnostic {
    Related(DiagnosticRelatedInformation),
    SuggestedFix(SuggestedFix),
    MessageLine(String),
}

fn map_rust_child_diagnostic(
    rd: &RustDiagnostic,
    workspace_root: &PathBuf,
) -> MappedRustChildDiagnostic {
    let span: &DiagnosticSpan = match rd.spans.iter().find(|s| s.is_primary) {
        Some(span) => span,
        None => {
            // `rustc` uses these spanless children as a way to print multi-line
            // messages
            return MappedRustChildDiagnostic::MessageLine(rd.message.clone());
        }
    };

    // If we have a primary span use its location, otherwise use the parent
    let location = map_span_to_location(&span, workspace_root);

    if let Some(suggested_replacement) = &span.suggested_replacement {
        // Include our replacement in the title unless it's empty
        let title = if !suggested_replacement.is_empty() {
            format!("{}: '{}'", rd.message, suggested_replacement)
        } else {
            rd.message.clone()
        };

        MappedRustChildDiagnostic::SuggestedFix(SuggestedFix {
            title,
            location,
            replacement: suggested_replacement.clone(),
            applicability: span.suggestion_applicability.clone().unwrap_or(Applicability::Unknown),
            diagnostics: vec![],
        })
    } else {
        MappedRustChildDiagnostic::Related(DiagnosticRelatedInformation {
            location,
            message: rd.message.clone(),
        })
    }
}

struct MappedRustDiagnostic {
    location: Location,
    diagnostic: Diagnostic,
    suggested_fixes: Vec<SuggestedFix>,
}

/// Converts a Rust root diagnostic to LSP form
///
/// This flattens the Rust diagnostic by:
///
/// 1. Creating a LSP diagnostic with the root message and primary span.
/// 2. Adding any labelled secondary spans to `relatedInformation`
/// 3. Categorising child diagnostics as either `SuggestedFix`es,
///    `relatedInformation` or additional message lines.
///
/// If the diagnostic has no primary span this will return `None`
fn map_rust_diagnostic_to_lsp(
    rd: &RustDiagnostic,
    workspace_root: &PathBuf,
) -> Option<MappedRustDiagnostic> {
    let primary_span = rd.spans.iter().find(|s| s.is_primary)?;

    let location = map_span_to_location(&primary_span, workspace_root);

    let severity = map_level_to_severity(rd.level);
    let mut primary_span_label = primary_span.label.as_ref();

    let mut source = String::from("rustc");
    let mut code = rd.code.as_ref().map(|c| c.code.clone());
    if let Some(code_val) = &code {
        // See if this is an RFC #2103 scoped lint (e.g. from Clippy)
        let scoped_code: Vec<&str> = code_val.split("::").collect();
        if scoped_code.len() == 2 {
            source = String::from(scoped_code[0]);
            code = Some(String::from(scoped_code[1]));
        }
    }

    let mut related_information = vec![];
    let mut tags = vec![];

    for secondary_span in rd.spans.iter().filter(|s| !s.is_primary) {
        let related = map_secondary_span_to_related(secondary_span, workspace_root);
        if let Some(related) = related {
            related_information.push(related);
        }
    }

    let mut suggested_fixes = vec![];
    let mut message = rd.message.clone();
    for child in &rd.children {
        let child = map_rust_child_diagnostic(&child, workspace_root);
        match child {
            MappedRustChildDiagnostic::Related(related) => related_information.push(related),
            MappedRustChildDiagnostic::SuggestedFix(suggested_fix) => {
                suggested_fixes.push(suggested_fix)
            }
            MappedRustChildDiagnostic::MessageLine(message_line) => {
                write!(&mut message, "\n{}", message_line).unwrap();

                // These secondary messages usually duplicate the content of the
                // primary span label.
                primary_span_label = None;
            }
        }
    }

    if let Some(primary_span_label) = primary_span_label {
        write!(&mut message, "\n{}", primary_span_label).unwrap();
    }

    if is_unused_or_unnecessary(rd) {
        tags.push(DiagnosticTag::Unnecessary);
    }

    if is_deprecated(rd) {
        tags.push(DiagnosticTag::Deprecated);
    }

    let diagnostic = Diagnostic {
        range: location.range,
        severity,
        code: code.map(NumberOrString::String),
        source: Some(source),
        message: rd.message.clone(),
        related_information: if !related_information.is_empty() {
            Some(related_information)
        } else {
            None
        },
        tags: if !tags.is_empty() { Some(tags) } else { None },
    };

    Some(MappedRustDiagnostic { location, diagnostic, suggested_fixes })
}

fn are_diagnostics_equal(left: &Diagnostic, right: &Diagnostic) -> bool {
    left.source == right.source
        && left.severity == right.severity
        && left.range == right.range
        && left.message == right.message
}
