use crate::world::Options;
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
    NumberOrString, Position, Range, Url, WorkDoneProgress, WorkDoneProgressBegin,
    WorkDoneProgressEnd, WorkDoneProgressReport,
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
    pub fn new(options: &Options, workspace_root: PathBuf) -> CheckWatcher {
        let check_command = options.cargo_check_command.clone();
        let check_args = options.cargo_check_args.clone();
        let shared = Arc::new(RwLock::new(CheckWatcherSharedState::new()));

        let (task_send, task_recv) = unbounded::<CheckTask>();
        let (cmd_send, cmd_recv) = unbounded::<CheckCommand>();
        let shared_ = shared.clone();
        let handle = std::thread::spawn(move || {
            let mut check =
                CheckWatcherState::new(check_command, check_args, workspace_root, shared_);
            check.run(&task_send, &cmd_recv);
        });

        CheckWatcher { task_recv, cmd_send, handle, shared }
    }

    pub fn update(&self) {
        self.cmd_send.send(CheckCommand::Update).unwrap();
    }
}

pub struct CheckWatcherState {
    check_command: Option<String>,
    check_args: Vec<String>,
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
    Status(WorkDoneProgress),
}

pub enum CheckCommand {
    Update,
}

impl CheckWatcherState {
    pub fn new(
        check_command: Option<String>,
        check_args: Vec<String>,
        workspace_root: PathBuf,
        shared: Arc<RwLock<CheckWatcherSharedState>>,
    ) -> CheckWatcherState {
        let watcher = WatchThread::new(check_command.as_ref(), &check_args, &workspace_root);
        CheckWatcherState {
            check_command,
            check_args,
            workspace_root,
            running: false,
            watcher,
            last_update_req: None,
            shared,
        }
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
                self.watcher = WatchThread::new(
                    self.check_command.as_ref(),
                    &self.check_args,
                    &self.workspace_root,
                );
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
struct WatchThread {
    message_recv: Receiver<CheckEvent>,
    cancel_send: Sender<()>,
}

enum CheckEvent {
    Begin,
    Msg(cargo_metadata::Message),
    End,
}

impl WatchThread {
    fn new(
        check_command: Option<&String>,
        check_args: &[String],
        workspace_root: &PathBuf,
    ) -> WatchThread {
        let check_command = check_command.cloned().unwrap_or("check".to_string());
        let mut args: Vec<String> = vec![
            check_command,
            "--message-format=json".to_string(),
            "--manifest-path".to_string(),
            format!("{}/Cargo.toml", workspace_root.to_string_lossy()),
        ];
        args.extend(check_args.iter().cloned());

        let (message_send, message_recv) = unbounded();
        let (cancel_send, cancel_recv) = unbounded();
        std::thread::spawn(move || {
            let mut command = Command::new("cargo")
                .args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("couldn't launch cargo");

            message_send.send(CheckEvent::Begin).unwrap();
            for message in cargo_metadata::parse_messages(command.stdout.take().unwrap()) {
                match cancel_recv.try_recv() {
                    Ok(()) | Err(TryRecvError::Disconnected) => {
                        command.kill().expect("couldn't kill command");
                    }
                    Err(TryRecvError::Empty) => (),
                }

                message_send.send(CheckEvent::Msg(message.unwrap())).unwrap();
            }
            message_send.send(CheckEvent::End).unwrap();
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

#[derive(Debug)]
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
        message,
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

#[cfg(test)]
mod test {
    use super::*;

    fn parse_diagnostic(val: &str) -> cargo_metadata::diagnostic::Diagnostic {
        serde_json::from_str::<cargo_metadata::diagnostic::Diagnostic>(val).unwrap()
    }

    #[test]
    fn snap_rustc_incompatible_type_for_trait() {
        let diag = parse_diagnostic(
            r##"{
                "message": "method `next` has an incompatible type for trait",
                "code": {
                    "code": "E0053",
                    "explanation": "\nThe parameters of any trait method must match between a trait implementation\nand the trait definition.\n\nHere are a couple examples of this error:\n\n```compile_fail,E0053\ntrait Foo {\n    fn foo(x: u16);\n    fn bar(&self);\n}\n\nstruct Bar;\n\nimpl Foo for Bar {\n    // error, expected u16, found i16\n    fn foo(x: i16) { }\n\n    // error, types differ in mutability\n    fn bar(&mut self) { }\n}\n```\n"
                },
                "level": "error",
                "spans": [
                    {
                        "file_name": "compiler/ty/list_iter.rs",
                        "byte_start": 1307,
                        "byte_end": 1350,
                        "line_start": 52,
                        "line_end": 52,
                        "column_start": 5,
                        "column_end": 48,
                        "is_primary": true,
                        "text": [
                            {
                                "text": "    fn next(&self) -> Option<&'list ty::Ref<M>> {",
                                "highlight_start": 5,
                                "highlight_end": 48
                            }
                        ],
                        "label": "types differ in mutability",
                        "suggested_replacement": null,
                        "suggestion_applicability": null,
                        "expansion": null
                    }
                ],
                "children": [
                    {
                        "message": "expected type `fn(&mut ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&ty::Ref<M>>`\n   found type `fn(&ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&'list ty::Ref<M>>`",
                        "code": null,
                        "level": "note",
                        "spans": [],
                        "children": [],
                        "rendered": null
                    }
                ],
                "rendered": "error[E0053]: method `next` has an incompatible type for trait\n  --> compiler/ty/list_iter.rs:52:5\n   |\n52 |     fn next(&self) -> Option<&'list ty::Ref<M>> {\n   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ types differ in mutability\n   |\n   = note: expected type `fn(&mut ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&ty::Ref<M>>`\n              found type `fn(&ty::list_iter::ListIterator<'list, M>) -> std::option::Option<&'list ty::Ref<M>>`\n\n"
            }
            "##,
        );

        let workspace_root = PathBuf::from("/test/");
        let diag =
            map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
        insta::assert_debug_snapshot!(diag);
    }

    #[test]
    fn snap_rustc_unused_variable() {
        let diag = parse_diagnostic(
            r##"{
    "message": "unused variable: `foo`",
    "code": {
        "code": "unused_variables",
        "explanation": null
    },
    "level": "warning",
    "spans": [
        {
            "file_name": "driver/subcommand/repl.rs",
            "byte_start": 9228,
            "byte_end": 9231,
            "line_start": 291,
            "line_end": 291,
            "column_start": 9,
            "column_end": 12,
            "is_primary": true,
            "text": [
                {
                    "text": "    let foo = 42;",
                    "highlight_start": 9,
                    "highlight_end": 12
                }
            ],
            "label": null,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [
        {
            "message": "#[warn(unused_variables)] on by default",
            "code": null,
            "level": "note",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "consider prefixing with an underscore",
            "code": null,
            "level": "help",
            "spans": [
                {
                    "file_name": "driver/subcommand/repl.rs",
                    "byte_start": 9228,
                    "byte_end": 9231,
                    "line_start": 291,
                    "line_end": 291,
                    "column_start": 9,
                    "column_end": 12,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    let foo = 42;",
                            "highlight_start": 9,
                            "highlight_end": 12
                        }
                    ],
                    "label": null,
                    "suggested_replacement": "_foo",
                    "suggestion_applicability": "MachineApplicable",
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        }
    ],
    "rendered": "warning: unused variable: `foo`\n   --> driver/subcommand/repl.rs:291:9\n    |\n291 |     let foo = 42;\n    |         ^^^ help: consider prefixing with an underscore: `_foo`\n    |\n    = note: #[warn(unused_variables)] on by default\n\n"
}"##,
        );

        let workspace_root = PathBuf::from("/test/");
        let diag =
            map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
        insta::assert_debug_snapshot!(diag);
    }

    #[test]
    fn snap_rustc_wrong_number_of_parameters() {
        let diag = parse_diagnostic(
            r##"{
    "message": "this function takes 2 parameters but 3 parameters were supplied",
    "code": {
        "code": "E0061",
        "explanation": "\nThe number of arguments passed to a function must match the number of arguments\nspecified in the function signature.\n\nFor example, a function like:\n\n```\nfn f(a: u16, b: &str) {}\n```\n\nMust always be called with exactly two arguments, e.g., `f(2, \"test\")`.\n\nNote that Rust does not have a notion of optional function arguments or\nvariadic functions (except for its C-FFI).\n"
    },
    "level": "error",
    "spans": [
        {
            "file_name": "compiler/ty/select.rs",
            "byte_start": 8787,
            "byte_end": 9241,
            "line_start": 219,
            "line_end": 231,
            "column_start": 5,
            "column_end": 6,
            "is_primary": false,
            "text": [
                {
                    "text": "    pub fn add_evidence(",
                    "highlight_start": 5,
                    "highlight_end": 25
                },
                {
                    "text": "        &mut self,",
                    "highlight_start": 1,
                    "highlight_end": 19
                },
                {
                    "text": "        target_poly: &ty::Ref<ty::Poly>,",
                    "highlight_start": 1,
                    "highlight_end": 41
                },
                {
                    "text": "        evidence_poly: &ty::Ref<ty::Poly>,",
                    "highlight_start": 1,
                    "highlight_end": 43
                },
                {
                    "text": "    ) {",
                    "highlight_start": 1,
                    "highlight_end": 8
                },
                {
                    "text": "        match target_poly {",
                    "highlight_start": 1,
                    "highlight_end": 28
                },
                {
                    "text": "            ty::Ref::Var(tvar, _) => self.add_var_evidence(tvar, evidence_poly),",
                    "highlight_start": 1,
                    "highlight_end": 81
                },
                {
                    "text": "            ty::Ref::Fixed(target_ty) => {",
                    "highlight_start": 1,
                    "highlight_end": 43
                },
                {
                    "text": "                let evidence_ty = evidence_poly.resolve_to_ty();",
                    "highlight_start": 1,
                    "highlight_end": 65
                },
                {
                    "text": "                self.add_evidence_ty(target_ty, evidence_poly, evidence_ty)",
                    "highlight_start": 1,
                    "highlight_end": 76
                },
                {
                    "text": "            }",
                    "highlight_start": 1,
                    "highlight_end": 14
                },
                {
                    "text": "        }",
                    "highlight_start": 1,
                    "highlight_end": 10
                },
                {
                    "text": "    }",
                    "highlight_start": 1,
                    "highlight_end": 6
                }
            ],
            "label": "defined here",
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        },
        {
            "file_name": "compiler/ty/select.rs",
            "byte_start": 4045,
            "byte_end": 4057,
            "line_start": 104,
            "line_end": 104,
            "column_start": 18,
            "column_end": 30,
            "is_primary": true,
            "text": [
                {
                    "text": "            self.add_evidence(target_fixed, evidence_fixed, false);",
                    "highlight_start": 18,
                    "highlight_end": 30
                }
            ],
            "label": "expected 2 parameters",
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [],
    "rendered": "error[E0061]: this function takes 2 parameters but 3 parameters were supplied\n   --> compiler/ty/select.rs:104:18\n    |\n104 |               self.add_evidence(target_fixed, evidence_fixed, false);\n    |                    ^^^^^^^^^^^^ expected 2 parameters\n...\n219 | /     pub fn add_evidence(\n220 | |         &mut self,\n221 | |         target_poly: &ty::Ref<ty::Poly>,\n222 | |         evidence_poly: &ty::Ref<ty::Poly>,\n...   |\n230 | |         }\n231 | |     }\n    | |_____- defined here\n\n"
}"##,
        );

        let workspace_root = PathBuf::from("/test/");
        let diag =
            map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
        insta::assert_debug_snapshot!(diag);
    }

    #[test]
    fn snap_clippy_pass_by_ref() {
        let diag = parse_diagnostic(
            r##"{
    "message": "this argument is passed by reference, but would be more efficient if passed by value",
    "code": {
        "code": "clippy::trivially_copy_pass_by_ref",
        "explanation": null
    },
    "level": "warning",
    "spans": [
        {
            "file_name": "compiler/mir/tagset.rs",
            "byte_start": 941,
            "byte_end": 946,
            "line_start": 42,
            "line_end": 42,
            "column_start": 24,
            "column_end": 29,
            "is_primary": true,
            "text": [
                {
                    "text": "    pub fn is_disjoint(&self, other: Self) -> bool {",
                    "highlight_start": 24,
                    "highlight_end": 29
                }
            ],
            "label": null,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [
        {
            "message": "lint level defined here",
            "code": null,
            "level": "note",
            "spans": [
                {
                    "file_name": "compiler/lib.rs",
                    "byte_start": 8,
                    "byte_end": 19,
                    "line_start": 1,
                    "line_end": 1,
                    "column_start": 9,
                    "column_end": 20,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "#![warn(clippy::all)]",
                            "highlight_start": 9,
                            "highlight_end": 20
                        }
                    ],
                    "label": null,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        },
        {
            "message": "#[warn(clippy::trivially_copy_pass_by_ref)] implied by #[warn(clippy::all)]",
            "code": null,
            "level": "note",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref",
            "code": null,
            "level": "help",
            "spans": [],
            "children": [],
            "rendered": null
        },
        {
            "message": "consider passing by value instead",
            "code": null,
            "level": "help",
            "spans": [
                {
                    "file_name": "compiler/mir/tagset.rs",
                    "byte_start": 941,
                    "byte_end": 946,
                    "line_start": 42,
                    "line_end": 42,
                    "column_start": 24,
                    "column_end": 29,
                    "is_primary": true,
                    "text": [
                        {
                            "text": "    pub fn is_disjoint(&self, other: Self) -> bool {",
                            "highlight_start": 24,
                            "highlight_end": 29
                        }
                    ],
                    "label": null,
                    "suggested_replacement": "self",
                    "suggestion_applicability": "Unspecified",
                    "expansion": null
                }
            ],
            "children": [],
            "rendered": null
        }
    ],
    "rendered": "warning: this argument is passed by reference, but would be more efficient if passed by value\n  --> compiler/mir/tagset.rs:42:24\n   |\n42 |     pub fn is_disjoint(&self, other: Self) -> bool {\n   |                        ^^^^^ help: consider passing by value instead: `self`\n   |\nnote: lint level defined here\n  --> compiler/lib.rs:1:9\n   |\n1  | #![warn(clippy::all)]\n   |         ^^^^^^^^^^^\n   = note: #[warn(clippy::trivially_copy_pass_by_ref)] implied by #[warn(clippy::all)]\n   = help: for further information visit https://rust-lang.github.io/rust-clippy/master/index.html#trivially_copy_pass_by_ref\n\n"
}"##,
        );

        let workspace_root = PathBuf::from("/test/");
        let diag =
            map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
        insta::assert_debug_snapshot!(diag);
    }

    #[test]
    fn snap_rustc_mismatched_type() {
        let diag = parse_diagnostic(
            r##"{
    "message": "mismatched types",
    "code": {
        "code": "E0308",
        "explanation": "\nThis error occurs when the compiler was unable to infer the concrete type of a\nvariable. It can occur for several cases, the most common of which is a\nmismatch in the expected type that the compiler inferred for a variable's\ninitializing expression, and the actual type explicitly assigned to the\nvariable.\n\nFor example:\n\n```compile_fail,E0308\nlet x: i32 = \"I am not a number!\";\n//     ~~~   ~~~~~~~~~~~~~~~~~~~~\n//      |             |\n//      |    initializing expression;\n//      |    compiler infers type `&str`\n//      |\n//    type `i32` assigned to variable `x`\n```\n"
    },
    "level": "error",
    "spans": [
        {
            "file_name": "runtime/compiler_support.rs",
            "byte_start": 1589,
            "byte_end": 1594,
            "line_start": 48,
            "line_end": 48,
            "column_start": 65,
            "column_end": 70,
            "is_primary": true,
            "text": [
                {
                    "text": "    let layout = alloc::Layout::from_size_align_unchecked(size, align);",
                    "highlight_start": 65,
                    "highlight_end": 70
                }
            ],
            "label": "expected usize, found u32",
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "expansion": null
        }
    ],
    "children": [],
    "rendered": "error[E0308]: mismatched types\n  --> runtime/compiler_support.rs:48:65\n   |\n48 |     let layout = alloc::Layout::from_size_align_unchecked(size, align);\n   |                                                                 ^^^^^ expected usize, found u32\n\n"
}"##,
        );

        let workspace_root = PathBuf::from("/test/");
        let diag =
            map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
        insta::assert_debug_snapshot!(diag);
    }

    #[test]
    fn snap_handles_macro_location() {
        let diag = parse_diagnostic(
            r##"{
    "rendered": "error[E0277]: can't compare `{integer}` with `&str`\n --> src/main.rs:2:5\n  |\n2 |     assert_eq!(1, \"love\");\n  |     ^^^^^^^^^^^^^^^^^^^^^^ no implementation for `{integer} == &str`\n  |\n  = help: the trait `std::cmp::PartialEq<&str>` is not implemented for `{integer}`\n  = note: this error originates in a macro outside of the current crate (in Nightly builds, run with -Z external-macro-backtrace for more info)\n\n",
    "children": [
        {
            "children": [],
            "code": null,
            "level": "help",
            "message": "the trait `std::cmp::PartialEq<&str>` is not implemented for `{integer}`",
            "rendered": null,
            "spans": []
        }
    ],
    "code": {
        "code": "E0277",
        "explanation": "\nYou tried to use a type which doesn't implement some trait in a place which\nexpected that trait. Erroneous code example:\n\n```compile_fail,E0277\n// here we declare the Foo trait with a bar method\ntrait Foo {\n    fn bar(&self);\n}\n\n// we now declare a function which takes an object implementing the Foo trait\nfn some_func<T: Foo>(foo: T) {\n    foo.bar();\n}\n\nfn main() {\n    // we now call the method with the i32 type, which doesn't implement\n    // the Foo trait\n    some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied\n}\n```\n\nIn order to fix this error, verify that the type you're using does implement\nthe trait. Example:\n\n```\ntrait Foo {\n    fn bar(&self);\n}\n\nfn some_func<T: Foo>(foo: T) {\n    foo.bar(); // we can now use this method since i32 implements the\n               // Foo trait\n}\n\n// we implement the trait on the i32 type\nimpl Foo for i32 {\n    fn bar(&self) {}\n}\n\nfn main() {\n    some_func(5i32); // ok!\n}\n```\n\nOr in a generic context, an erroneous code example would look like:\n\n```compile_fail,E0277\nfn some_func<T>(foo: T) {\n    println!(\"{:?}\", foo); // error: the trait `core::fmt::Debug` is not\n                           //        implemented for the type `T`\n}\n\nfn main() {\n    // We now call the method with the i32 type,\n    // which *does* implement the Debug trait.\n    some_func(5i32);\n}\n```\n\nNote that the error here is in the definition of the generic function: Although\nwe only call it with a parameter that does implement `Debug`, the compiler\nstill rejects the function: It must work with all possible input types. In\norder to make this example compile, we need to restrict the generic type we're\naccepting:\n\n```\nuse std::fmt;\n\n// Restrict the input type to types that implement Debug.\nfn some_func<T: fmt::Debug>(foo: T) {\n    println!(\"{:?}\", foo);\n}\n\nfn main() {\n    // Calling the method is still fine, as i32 implements Debug.\n    some_func(5i32);\n\n    // This would fail to compile now:\n    // struct WithoutDebug;\n    // some_func(WithoutDebug);\n}\n```\n\nRust only looks at the signature of the called function, as such it must\nalready specify all requirements that will be used for every type parameter.\n"
    },
    "level": "error",
    "message": "can't compare `{integer}` with `&str`",
    "spans": [
        {
            "byte_end": 155,
            "byte_start": 153,
            "column_end": 33,
            "column_start": 31,
            "expansion": {
                "def_site_span": {
                    "byte_end": 940,
                    "byte_start": 0,
                    "column_end": 6,
                    "column_start": 1,
                    "expansion": null,
                    "file_name": "<::core::macros::assert_eq macros>",
                    "is_primary": false,
                    "label": null,
                    "line_end": 36,
                    "line_start": 1,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                            "highlight_end": 35,
                            "highlight_start": 1,
                            "text": "($ left : expr, $ right : expr) =>"
                        },
                        {
                            "highlight_end": 3,
                            "highlight_start": 1,
                            "text": "({"
                        },
                        {
                            "highlight_end": 33,
                            "highlight_start": 1,
                            "text": "     match (& $ left, & $ right)"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     {"
                        },
                        {
                            "highlight_end": 34,
                            "highlight_start": 1,
                            "text": "         (left_val, right_val) =>"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         {"
                        },
                        {
                            "highlight_end": 46,
                            "highlight_start": 1,
                            "text": "             if ! (* left_val == * right_val)"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             {"
                        },
                        {
                            "highlight_end": 25,
                            "highlight_start": 1,
                            "text": "                 panic !"
                        },
                        {
                            "highlight_end": 57,
                            "highlight_start": 1,
                            "text": "                 (r#\"assertion failed: `(left == right)`"
                        },
                        {
                            "highlight_end": 16,
                            "highlight_start": 1,
                            "text": "  left: `{:?}`,"
                        },
                        {
                            "highlight_end": 18,
                            "highlight_start": 1,
                            "text": " right: `{:?}`\"#,"
                        },
                        {
                            "highlight_end": 47,
                            "highlight_start": 1,
                            "text": "                  & * left_val, & * right_val)"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             }"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         }"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     }"
                        },
                        {
                            "highlight_end": 42,
                            "highlight_start": 1,
                            "text": " }) ; ($ left : expr, $ right : expr,) =>"
                        },
                        {
                            "highlight_end": 49,
                            "highlight_start": 1,
                            "text": "({ $ crate :: assert_eq ! ($ left, $ right) }) ;"
                        },
                        {
                            "highlight_end": 53,
                            "highlight_start": 1,
                            "text": "($ left : expr, $ right : expr, $ ($ arg : tt) +) =>"
                        },
                        {
                            "highlight_end": 3,
                            "highlight_start": 1,
                            "text": "({"
                        },
                        {
                            "highlight_end": 37,
                            "highlight_start": 1,
                            "text": "     match (& ($ left), & ($ right))"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     {"
                        },
                        {
                            "highlight_end": 34,
                            "highlight_start": 1,
                            "text": "         (left_val, right_val) =>"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         {"
                        },
                        {
                            "highlight_end": 46,
                            "highlight_start": 1,
                            "text": "             if ! (* left_val == * right_val)"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             {"
                        },
                        {
                            "highlight_end": 25,
                            "highlight_start": 1,
                            "text": "                 panic !"
                        },
                        {
                            "highlight_end": 57,
                            "highlight_start": 1,
                            "text": "                 (r#\"assertion failed: `(left == right)`"
                        },
                        {
                            "highlight_end": 16,
                            "highlight_start": 1,
                            "text": "  left: `{:?}`,"
                        },
                        {
                            "highlight_end": 22,
                            "highlight_start": 1,
                            "text": " right: `{:?}`: {}\"#,"
                        },
                        {
                            "highlight_end": 72,
                            "highlight_start": 1,
                            "text": "                  & * left_val, & * right_val, $ crate :: format_args !"
                        },
                        {
                            "highlight_end": 33,
                            "highlight_start": 1,
                            "text": "                  ($ ($ arg) +))"
                        },
                        {
                            "highlight_end": 15,
                            "highlight_start": 1,
                            "text": "             }"
                        },
                        {
                            "highlight_end": 11,
                            "highlight_start": 1,
                            "text": "         }"
                        },
                        {
                            "highlight_end": 7,
                            "highlight_start": 1,
                            "text": "     }"
                        },
                        {
                            "highlight_end": 6,
                            "highlight_start": 1,
                            "text": " }) ;"
                        }
                    ]
                },
                "macro_decl_name": "assert_eq!",
                "span": {
                    "byte_end": 38,
                    "byte_start": 16,
                    "column_end": 27,
                    "column_start": 5,
                    "expansion": null,
                    "file_name": "src/main.rs",
                    "is_primary": false,
                    "label": null,
                    "line_end": 2,
                    "line_start": 2,
                    "suggested_replacement": null,
                    "suggestion_applicability": null,
                    "text": [
                        {
                            "highlight_end": 27,
                            "highlight_start": 5,
                            "text": "    assert_eq!(1, \"love\");"
                        }
                    ]
                }
            },
            "file_name": "<::core::macros::assert_eq macros>",
            "is_primary": true,
            "label": "no implementation for `{integer} == &str`",
            "line_end": 7,
            "line_start": 7,
            "suggested_replacement": null,
            "suggestion_applicability": null,
            "text": [
                {
                    "highlight_end": 33,
                    "highlight_start": 31,
                    "text": "             if ! (* left_val == * right_val)"
                }
            ]
        }
    ]
}"##,
        );

        let workspace_root = PathBuf::from("/test/");
        let diag =
            map_rust_diagnostic_to_lsp(&diag, &workspace_root).expect("couldn't map diagnostic");
        insta::assert_debug_snapshot!(diag);
    }
}
