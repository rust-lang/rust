//! The main loop of `rust-analyzer` responsible for dispatching LSP
//! requests/replies and notifications back to the client.

mod handlers;
pub(crate) mod request_metrics;

use std::{
    env,
    error::Error,
    fmt,
    ops::Range,
    panic,
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam_channel::{never, select, unbounded, RecvError, Sender};
use lsp_server::{
    Connection, ErrorCode, Message, Notification, ReqQueue, Request, RequestId, Response,
};
use lsp_types::{request::Request as _, NumberOrString, TextDocumentContentChangeEvent};
use ra_flycheck::CheckTask;
use ra_ide::{Canceled, FileId, LineIndex};
use ra_prof::profile;
use ra_project_model::{PackageRoot, ProjectWorkspace};
use rustc_hash::FxHashSet;
use serde::{de::DeserializeOwned, Serialize};
use threadpool::ThreadPool;

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    diagnostics::DiagnosticTask,
    from_proto,
    global_state::{file_id_to_url, GlobalState, GlobalStateSnapshot},
    lsp_ext,
    main_loop::request_metrics::RequestMetrics,
    Result,
};
use ra_db::VfsPath;

#[derive(Debug)]
pub struct LspError {
    pub code: i32,
    pub message: String,
}

impl LspError {
    pub const UNKNOWN_FILE: i32 = -32900;

    pub fn new(code: i32, message: String) -> LspError {
        LspError { code, message }
    }
}

impl fmt::Display for LspError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Language Server request failed with {}. ({})", self.code, self.message)
    }
}

impl Error for LspError {}

pub fn main_loop(config: Config, connection: Connection) -> Result<()> {
    log::info!("initial config: {:#?}", config);

    // Windows scheduler implements priority boosts: if thread waits for an
    // event (like a condvar), and event fires, priority of the thread is
    // temporary bumped. This optimization backfires in our case: each time the
    // `main_loop` schedules a task to run on a threadpool, the worker threads
    // gets a higher priority, and (on a machine with fewer cores) displaces the
    // main loop! We work-around this by marking the main loop as a
    // higher-priority thread.
    //
    // https://docs.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
    // https://docs.microsoft.com/en-us/windows/win32/procthread/priority-boosts
    // https://github.com/rust-analyzer/rust-analyzer/issues/2835
    #[cfg(windows)]
    unsafe {
        use winapi::um::processthreadsapi::*;
        let thread = GetCurrentThread();
        let thread_priority_above_normal = 1;
        SetThreadPriority(thread, thread_priority_above_normal);
    }

    let mut loop_state = LoopState::default();
    let mut global_state = {
        let workspaces = {
            if config.linked_projects.is_empty() && config.notifications.cargo_toml_not_found {
                show_message(
                    lsp_types::MessageType::Error,
                    "rust-analyzer failed to discover workspace".to_string(),
                    &connection.sender,
                );
            };

            config
                .linked_projects
                .iter()
                .filter_map(|project| match project {
                    LinkedProject::ProjectManifest(manifest) => {
                        ra_project_model::ProjectWorkspace::load(
                            manifest.clone(),
                            &config.cargo,
                            config.with_sysroot,
                        )
                        .map_err(|err| {
                            log::error!("failed to load workspace: {:#}", err);
                            show_message(
                                lsp_types::MessageType::Error,
                                format!("rust-analyzer failed to load workspace: {:#}", err),
                                &connection.sender,
                            );
                        })
                        .ok()
                    }
                    LinkedProject::InlineJsonProject(it) => {
                        Some(ra_project_model::ProjectWorkspace::Json {
                            project: it.clone(),
                            project_location: config.root_path.clone(),
                        })
                    }
                })
                .collect::<Vec<_>>()
        };

        if let FilesWatcher::Client = config.files.watcher {
            let registration_options = lsp_types::DidChangeWatchedFilesRegistrationOptions {
                watchers: workspaces
                    .iter()
                    .flat_map(ProjectWorkspace::to_roots)
                    .filter(PackageRoot::is_member)
                    .map(|root| format!("{}/**/*.rs", root.path().display()))
                    .map(|glob_pattern| lsp_types::FileSystemWatcher { glob_pattern, kind: None })
                    .collect(),
            };
            let registration = lsp_types::Registration {
                id: "file-watcher".to_string(),
                method: "workspace/didChangeWatchedFiles".to_string(),
                register_options: Some(serde_json::to_value(registration_options).unwrap()),
            };
            let params = lsp_types::RegistrationParams { registrations: vec![registration] };
            let request = loop_state.req_queue.outgoing.register(
                lsp_types::request::RegisterCapability::METHOD.to_string(),
                params,
                DO_NOTHING,
            );
            connection.sender.send(request.into()).unwrap();
        }

        GlobalState::new(workspaces, config.lru_capacity, config)
    };

    let pool = ThreadPool::default();
    let (task_sender, task_receiver) = unbounded::<Task>();

    log::info!("server initialized, serving requests");
    {
        let task_sender = task_sender;
        loop {
            log::trace!("selecting");
            let event = select! {
                recv(&connection.receiver) -> msg => match msg {
                    Ok(msg) => Event::Msg(msg),
                    Err(RecvError) => return Err("client exited without shutdown".into()),
                },
                recv(task_receiver) -> task => Event::Task(task.unwrap()),
                recv(global_state.task_receiver) -> task => match task {
                    Ok(task) => Event::Vfs(task),
                    Err(RecvError) => return Err("vfs died".into()),
                },
                recv(global_state.flycheck.as_ref().map_or(&never(), |it| &it.task_recv)) -> task => match task {
                    Ok(task) => Event::CheckWatcher(task),
                    Err(RecvError) => return Err("check watcher died".into()),
                }
            };
            if let Event::Msg(Message::Request(req)) = &event {
                if connection.handle_shutdown(&req)? {
                    break;
                };
            }
            assert!(!global_state.vfs.read().0.has_changes());
            loop_turn(&pool, &task_sender, &connection, &mut global_state, &mut loop_state, event)?;
            assert!(!global_state.vfs.read().0.has_changes());
        }
    }
    global_state.analysis_host.request_cancellation();
    log::info!("waiting for tasks to finish...");
    task_receiver.into_iter().for_each(|task| {
        on_task(task, &connection.sender, &mut loop_state.req_queue.incoming, &mut global_state)
    });
    log::info!("...tasks have finished");
    log::info!("joining threadpool...");
    pool.join();
    drop(pool);
    log::info!("...threadpool has finished");

    let vfs = Arc::try_unwrap(global_state.vfs).expect("all snapshots should be dead");
    drop(vfs);

    Ok(())
}

#[derive(Debug)]
enum Task {
    Respond(Response),
    Notify(Notification),
    Diagnostic(DiagnosticTask),
}

enum Event {
    Msg(Message),
    Task(Task),
    Vfs(vfs::loader::Message),
    CheckWatcher(CheckTask),
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let debug_verbose_not = |not: &Notification, f: &mut fmt::Formatter| {
            f.debug_struct("Notification").field("method", &not.method).finish()
        };

        match self {
            Event::Msg(Message::Notification(not)) => {
                if notification_is::<lsp_types::notification::DidOpenTextDocument>(not)
                    || notification_is::<lsp_types::notification::DidChangeTextDocument>(not)
                {
                    return debug_verbose_not(not, f);
                }
            }
            Event::Task(Task::Notify(not)) => {
                if notification_is::<lsp_types::notification::PublishDiagnostics>(not) {
                    return debug_verbose_not(not, f);
                }
            }
            Event::Task(Task::Respond(resp)) => {
                return f
                    .debug_struct("Response")
                    .field("id", &resp.id)
                    .field("error", &resp.error)
                    .finish();
            }
            _ => (),
        }
        match self {
            Event::Msg(it) => fmt::Debug::fmt(it, f),
            Event::Task(it) => fmt::Debug::fmt(it, f),
            Event::Vfs(it) => fmt::Debug::fmt(it, f),
            Event::CheckWatcher(it) => fmt::Debug::fmt(it, f),
        }
    }
}

type ReqHandler = fn(&mut GlobalState, Response);
const DO_NOTHING: ReqHandler = |_, _| ();
type Incoming = lsp_server::Incoming<(&'static str, Instant)>;

#[derive(Default)]
struct LoopState {
    req_queue: ReqQueue<(&'static str, Instant), ReqHandler>,
    mem_docs: FxHashSet<VfsPath>,
    status: Status,
}

#[derive(Eq, PartialEq)]
enum Status {
    Loading,
    Ready,
}

impl Default for Status {
    fn default() -> Self {
        Status::Loading
    }
}

fn loop_turn(
    pool: &ThreadPool,
    task_sender: &Sender<Task>,
    connection: &Connection,
    global_state: &mut GlobalState,
    loop_state: &mut LoopState,
    event: Event,
) -> Result<()> {
    let loop_start = Instant::now();

    // NOTE: don't count blocking select! call as a loop-turn time
    let _p = profile("main_loop_inner/loop-turn");
    log::info!("loop turn = {:?}", event);
    let queue_count = pool.queued_count();
    if queue_count > 0 {
        log::info!("queued count = {}", queue_count);
    }

    let mut became_ready = false;
    match event {
        Event::Task(task) => {
            on_task(task, &connection.sender, &mut loop_state.req_queue.incoming, global_state);
            global_state.maybe_collect_garbage();
        }
        Event::Vfs(task) => match task {
            vfs::loader::Message::Loaded { files } => {
                let vfs = &mut global_state.vfs.write().0;
                for (path, contents) in files {
                    let path = VfsPath::from(path);
                    if !loop_state.mem_docs.contains(&path) {
                        vfs.set_file_contents(path, contents)
                    }
                }
            }
            vfs::loader::Message::Progress { n_entries_total, n_entries_done } => {
                if n_entries_done == n_entries_done {
                    loop_state.status = Status::Ready;
                    became_ready = true;
                }
                report_progress(
                    loop_state,
                    &connection.sender,
                    n_entries_done,
                    n_entries_total,
                    "roots scanned",
                )
            }
        },
        Event::CheckWatcher(task) => on_check_task(task, global_state, task_sender)?,
        Event::Msg(msg) => match msg {
            Message::Request(req) => on_request(
                global_state,
                &mut loop_state.req_queue.incoming,
                pool,
                task_sender,
                &connection.sender,
                loop_start,
                req,
            )?,
            Message::Notification(not) => {
                on_notification(&connection.sender, global_state, loop_state, not)?;
            }
            Message::Response(resp) => {
                let handler = loop_state.req_queue.outgoing.complete(resp.id.clone());
                handler(global_state, resp)
            }
        },
    };

    let state_changed = global_state.process_changes();

    if became_ready {
        if let Some(flycheck) = &global_state.flycheck {
            flycheck.update();
        }
    }

    if loop_state.status == Status::Ready && (state_changed || became_ready) {
        let subscriptions = loop_state
            .mem_docs
            .iter()
            .map(|path| global_state.vfs.read().0.file_id(&path).unwrap())
            .collect::<Vec<_>>();

        update_file_notifications_on_threadpool(
            pool,
            global_state.snapshot(),
            task_sender.clone(),
            subscriptions.clone(),
        );
        pool.execute({
            let subs = subscriptions;
            let snap = global_state.snapshot();
            move || snap.analysis().prime_caches(subs).unwrap_or_else(|_: Canceled| ())
        });
    }

    let loop_duration = loop_start.elapsed();
    if loop_duration > Duration::from_millis(100) {
        log::error!("overly long loop turn: {:?}", loop_duration);
        if env::var("RA_PROFILE").is_ok() {
            show_message(
                lsp_types::MessageType::Error,
                format!("overly long loop turn: {:?}", loop_duration),
                &connection.sender,
            );
        }
    }

    Ok(())
}

fn on_task(
    task: Task,
    msg_sender: &Sender<Message>,
    incoming_requests: &mut Incoming,
    state: &mut GlobalState,
) {
    match task {
        Task::Respond(response) => {
            if let Some((method, start)) = incoming_requests.complete(response.id.clone()) {
                let duration = start.elapsed();
                log::info!("handled req#{} in {:?}", response.id, duration);
                state.complete_request(RequestMetrics {
                    id: response.id.clone(),
                    method: method.to_string(),
                    duration,
                });
                msg_sender.send(response.into()).unwrap();
            }
        }
        Task::Notify(n) => {
            msg_sender.send(n.into()).unwrap();
        }
        Task::Diagnostic(task) => on_diagnostic_task(task, msg_sender, state),
    }
}

fn on_request(
    global_state: &mut GlobalState,
    incoming_requests: &mut Incoming,
    pool: &ThreadPool,
    task_sender: &Sender<Task>,
    msg_sender: &Sender<Message>,
    request_received: Instant,
    req: Request,
) -> Result<()> {
    let mut pool_dispatcher = PoolDispatcher {
        req: Some(req),
        pool,
        global_state,
        task_sender,
        msg_sender,
        incoming_requests,
        request_received,
    };
    pool_dispatcher
        .on_sync::<lsp_ext::CollectGarbage>(|s, ()| Ok(s.collect_garbage()))?
        .on_sync::<lsp_ext::JoinLines>(|s, p| handlers::handle_join_lines(s.snapshot(), p))?
        .on_sync::<lsp_ext::OnEnter>(|s, p| handlers::handle_on_enter(s.snapshot(), p))?
        .on_sync::<lsp_types::request::SelectionRangeRequest>(|s, p| {
            handlers::handle_selection_range(s.snapshot(), p)
        })?
        .on_sync::<lsp_ext::MatchingBrace>(|s, p| handlers::handle_matching_brace(s.snapshot(), p))?
        .on::<lsp_ext::AnalyzerStatus>(handlers::handle_analyzer_status)?
        .on::<lsp_ext::SyntaxTree>(handlers::handle_syntax_tree)?
        .on::<lsp_ext::ExpandMacro>(handlers::handle_expand_macro)?
        .on::<lsp_ext::ParentModule>(handlers::handle_parent_module)?
        .on::<lsp_ext::Runnables>(handlers::handle_runnables)?
        .on::<lsp_ext::InlayHints>(handlers::handle_inlay_hints)?
        .on::<lsp_ext::CodeActionRequest>(handlers::handle_code_action)?
        .on::<lsp_ext::ResolveCodeActionRequest>(handlers::handle_resolve_code_action)?
        .on::<lsp_ext::HoverRequest>(handlers::handle_hover)?
        .on::<lsp_types::request::OnTypeFormatting>(handlers::handle_on_type_formatting)?
        .on::<lsp_types::request::DocumentSymbolRequest>(handlers::handle_document_symbol)?
        .on::<lsp_types::request::WorkspaceSymbol>(handlers::handle_workspace_symbol)?
        .on::<lsp_types::request::GotoDefinition>(handlers::handle_goto_definition)?
        .on::<lsp_types::request::GotoImplementation>(handlers::handle_goto_implementation)?
        .on::<lsp_types::request::GotoTypeDefinition>(handlers::handle_goto_type_definition)?
        .on::<lsp_types::request::Completion>(handlers::handle_completion)?
        .on::<lsp_types::request::CodeLensRequest>(handlers::handle_code_lens)?
        .on::<lsp_types::request::CodeLensResolve>(handlers::handle_code_lens_resolve)?
        .on::<lsp_types::request::FoldingRangeRequest>(handlers::handle_folding_range)?
        .on::<lsp_types::request::SignatureHelpRequest>(handlers::handle_signature_help)?
        .on::<lsp_types::request::PrepareRenameRequest>(handlers::handle_prepare_rename)?
        .on::<lsp_types::request::Rename>(handlers::handle_rename)?
        .on::<lsp_types::request::References>(handlers::handle_references)?
        .on::<lsp_types::request::Formatting>(handlers::handle_formatting)?
        .on::<lsp_types::request::DocumentHighlightRequest>(handlers::handle_document_highlight)?
        .on::<lsp_types::request::CallHierarchyPrepare>(handlers::handle_call_hierarchy_prepare)?
        .on::<lsp_types::request::CallHierarchyIncomingCalls>(
            handlers::handle_call_hierarchy_incoming,
        )?
        .on::<lsp_types::request::CallHierarchyOutgoingCalls>(
            handlers::handle_call_hierarchy_outgoing,
        )?
        .on::<lsp_types::request::SemanticTokensRequest>(handlers::handle_semantic_tokens)?
        .on::<lsp_types::request::SemanticTokensRangeRequest>(
            handlers::handle_semantic_tokens_range,
        )?
        .on::<lsp_ext::Ssr>(handlers::handle_ssr)?
        .finish();
    Ok(())
}

fn on_notification(
    msg_sender: &Sender<Message>,
    global_state: &mut GlobalState,
    loop_state: &mut LoopState,
    not: Notification,
) -> Result<()> {
    let not = match notification_cast::<lsp_types::notification::Cancel>(not) {
        Ok(params) => {
            let id: RequestId = match params.id {
                NumberOrString::Number(id) => id.into(),
                NumberOrString::String(id) => id.into(),
            };
            if let Some(response) = loop_state.req_queue.incoming.cancel(id) {
                msg_sender.send(response.into()).unwrap()
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<lsp_types::notification::DidOpenTextDocument>(not) {
        Ok(params) => {
            if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                if !loop_state.mem_docs.insert(path.clone()) {
                    log::error!("duplicate DidOpenTextDocument: {}", path)
                }
                global_state
                    .vfs
                    .write()
                    .0
                    .set_file_contents(path, Some(params.text_document.text.into_bytes()));
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<lsp_types::notification::DidChangeTextDocument>(not) {
        Ok(params) => {
            if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                assert!(loop_state.mem_docs.contains(&path));
                let vfs = &mut global_state.vfs.write().0;
                let file_id = vfs.file_id(&path).unwrap();
                let mut text = String::from_utf8(vfs.file_contents(file_id).to_vec()).unwrap();
                apply_document_changes(&mut text, params.content_changes);
                vfs.set_file_contents(path, Some(text.into_bytes()))
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<lsp_types::notification::DidCloseTextDocument>(not) {
        Ok(params) => {
            if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                if !loop_state.mem_docs.remove(&path) {
                    log::error!("orphan DidCloseTextDocument: {}", path)
                }
                if let Some(path) = path.as_path() {
                    global_state.loader.invalidate(path.to_path_buf());
                }
            }
            let params = lsp_types::PublishDiagnosticsParams {
                uri: params.text_document.uri,
                diagnostics: Vec::new(),
                version: None,
            };
            let not = notification_new::<lsp_types::notification::PublishDiagnostics>(params);
            msg_sender.send(not.into()).unwrap();
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<lsp_types::notification::DidSaveTextDocument>(not) {
        Ok(_params) => {
            if let Some(flycheck) = &global_state.flycheck {
                flycheck.update();
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<lsp_types::notification::DidChangeConfiguration>(not) {
        Ok(_) => {
            // As stated in https://github.com/microsoft/language-server-protocol/issues/676,
            // this notification's parameters should be ignored and the actual config queried separately.
            let request = loop_state.req_queue.outgoing.register(
                lsp_types::request::WorkspaceConfiguration::METHOD.to_string(),
                lsp_types::ConfigurationParams {
                    items: vec![lsp_types::ConfigurationItem {
                        scope_uri: None,
                        section: Some("rust-analyzer".to_string()),
                    }],
                },
                |global_state, resp| {
                    log::debug!("config update response: '{:?}", resp);
                    let Response { error, result, .. } = resp;

                    match (error, result) {
                        (Some(err), _) => {
                            log::error!("failed to fetch the server settings: {:?}", err)
                        }
                        (None, Some(configs)) => {
                            if let Some(new_config) = configs.get(0) {
                                let mut config = global_state.config.clone();
                                config.update(&new_config);
                                global_state.update_configuration(config);
                            }
                        }
                        (None, None) => {
                            log::error!("received empty server settings response from the client")
                        }
                    }
                },
            );
            msg_sender.send(request.into())?;

            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<lsp_types::notification::DidChangeWatchedFiles>(not) {
        Ok(params) => {
            for change in params.changes {
                if let Ok(path) = from_proto::abs_path(&change.uri) {
                    global_state.loader.invalidate(path)
                }
            }
            return Ok(());
        }
        Err(not) => not,
    };
    if not.method.starts_with("$/") {
        return Ok(());
    }
    log::error!("unhandled notification: {:?}", not);
    Ok(())
}

fn apply_document_changes(
    old_text: &mut String,
    content_changes: Vec<TextDocumentContentChangeEvent>,
) {
    let mut line_index = LineIndex::new(old_text);
    // The changes we got must be applied sequentially, but can cross lines so we
    // have to keep our line index updated.
    // Some clients (e.g. Code) sort the ranges in reverse. As an optimization, we
    // remember the last valid line in the index and only rebuild it if needed.
    // The VFS will normalize the end of lines to `\n`.
    enum IndexValid {
        All,
        UpToLineExclusive(u64),
    }

    impl IndexValid {
        fn covers(&self, line: u64) -> bool {
            match *self {
                IndexValid::UpToLineExclusive(to) => to > line,
                _ => true,
            }
        }
    }

    let mut index_valid = IndexValid::All;
    for change in content_changes {
        match change.range {
            Some(range) => {
                if !index_valid.covers(range.end.line) {
                    line_index = LineIndex::new(&old_text);
                }
                index_valid = IndexValid::UpToLineExclusive(range.start.line);
                let range = from_proto::text_range(&line_index, range);
                old_text.replace_range(Range::<usize>::from(range), &change.text);
            }
            None => {
                *old_text = change.text;
                index_valid = IndexValid::UpToLineExclusive(0);
            }
        }
    }
}

fn on_check_task(
    task: CheckTask,
    global_state: &mut GlobalState,
    task_sender: &Sender<Task>,
) -> Result<()> {
    match task {
        CheckTask::ClearDiagnostics => {
            task_sender.send(Task::Diagnostic(DiagnosticTask::ClearCheck))?;
        }

        CheckTask::AddDiagnostic { workspace_root, diagnostic } => {
            let diagnostics = crate::diagnostics::to_proto::map_rust_diagnostic_to_lsp(
                &global_state.config.diagnostics,
                &diagnostic,
                &workspace_root,
            );
            for diag in diagnostics {
                let path = from_proto::vfs_path(&diag.location.uri)?;
                let file_id = match global_state.vfs.read().0.file_id(&path) {
                    Some(file) => FileId(file.0),
                    None => {
                        log::error!("File with cargo diagnostic not found in VFS: {}", path);
                        return Ok(());
                    }
                };

                task_sender.send(Task::Diagnostic(DiagnosticTask::AddCheck(
                    file_id,
                    diag.diagnostic,
                    diag.fixes.into_iter().map(|it| it.into()).collect(),
                )))?;
            }
        }

        CheckTask::Status(status) => {
            if global_state.config.client_caps.work_done_progress {
                let progress = match status {
                    ra_flycheck::Status::Being => {
                        lsp_types::WorkDoneProgress::Begin(lsp_types::WorkDoneProgressBegin {
                            title: "Running `cargo check`".to_string(),
                            cancellable: Some(false),
                            message: None,
                            percentage: None,
                        })
                    }
                    ra_flycheck::Status::Progress(target) => {
                        lsp_types::WorkDoneProgress::Report(lsp_types::WorkDoneProgressReport {
                            cancellable: Some(false),
                            message: Some(target),
                            percentage: None,
                        })
                    }
                    ra_flycheck::Status::End => {
                        lsp_types::WorkDoneProgress::End(lsp_types::WorkDoneProgressEnd {
                            message: None,
                        })
                    }
                };

                let params = lsp_types::ProgressParams {
                    token: lsp_types::ProgressToken::String(
                        "rustAnalyzer/cargoWatcher".to_string(),
                    ),
                    value: lsp_types::ProgressParamsValue::WorkDone(progress),
                };
                let not = notification_new::<lsp_types::notification::Progress>(params);
                task_sender.send(Task::Notify(not)).unwrap();
            }
        }
    };

    Ok(())
}

fn on_diagnostic_task(task: DiagnosticTask, msg_sender: &Sender<Message>, state: &mut GlobalState) {
    let subscriptions = state.diagnostics.handle_task(task);

    for file_id in subscriptions {
        let url = file_id_to_url(&state.vfs.read().0, file_id);
        let diagnostics = state.diagnostics.diagnostics_for(file_id).cloned().collect();
        let params = lsp_types::PublishDiagnosticsParams { uri: url, diagnostics, version: None };
        let not = notification_new::<lsp_types::notification::PublishDiagnostics>(params);
        msg_sender.send(not.into()).unwrap();
    }
}

fn report_progress(
    loop_state: &mut LoopState,
    sender: &Sender<Message>,
    done: usize,
    total: usize,
    message: &str,
) {
    let token = lsp_types::ProgressToken::String(format!("rustAnalyzer/{}", message));
    let message = Some(format!("{}/{} {}", done, total, message));
    let percentage = Some(100.0 * done as f64 / total.max(1) as f64);
    let work_done_progress = if done == 0 {
        let work_done_progress_create = loop_state.req_queue.outgoing.register(
            lsp_types::request::WorkDoneProgressCreate::METHOD.to_string(),
            lsp_types::WorkDoneProgressCreateParams { token: token.clone() },
            DO_NOTHING,
        );
        sender.send(work_done_progress_create.into()).unwrap();

        lsp_types::WorkDoneProgress::Begin(lsp_types::WorkDoneProgressBegin {
            title: "rust-analyzer".into(),
            cancellable: None,
            message,
            percentage,
        })
    } else if done < total {
        lsp_types::WorkDoneProgress::Report(lsp_types::WorkDoneProgressReport {
            cancellable: None,
            message,
            percentage,
        })
    } else {
        assert!(done == total);
        lsp_types::WorkDoneProgress::End(lsp_types::WorkDoneProgressEnd { message })
    };
    let notification =
        notification_new::<lsp_types::notification::Progress>(lsp_types::ProgressParams {
            token,
            value: lsp_types::ProgressParamsValue::WorkDone(work_done_progress),
        });
    sender.send(notification.into()).unwrap();
}

struct PoolDispatcher<'a> {
    req: Option<Request>,
    pool: &'a ThreadPool,
    global_state: &'a mut GlobalState,
    incoming_requests: &'a mut Incoming,
    msg_sender: &'a Sender<Message>,
    task_sender: &'a Sender<Task>,
    request_received: Instant,
}

impl<'a> PoolDispatcher<'a> {
    /// Dispatches the request onto the current thread
    fn on_sync<R>(
        &mut self,
        f: fn(&mut GlobalState, R::Params) -> Result<R::Result>,
    ) -> Result<&mut Self>
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + panic::UnwindSafe + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => {
                return Ok(self);
            }
        };
        let world = panic::AssertUnwindSafe(&mut *self.global_state);
        let task = panic::catch_unwind(move || {
            let result = f(world.0, params);
            result_to_task::<R>(id, result)
        })
        .map_err(|_| format!("sync task {:?} panicked", R::METHOD))?;
        on_task(task, self.msg_sender, self.incoming_requests, self.global_state);
        Ok(self)
    }

    /// Dispatches the request onto thread pool
    fn on<R>(
        &mut self,
        f: fn(GlobalStateSnapshot, R::Params) -> Result<R::Result>,
    ) -> Result<&mut Self>
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + Send + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => {
                return Ok(self);
            }
        };

        self.pool.execute({
            let world = self.global_state.snapshot();
            let sender = self.task_sender.clone();
            move || {
                let result = f(world, params);
                let task = result_to_task::<R>(id, result);
                sender.send(task).unwrap();
            }
        });

        Ok(self)
    }

    fn parse<R>(&mut self) -> Option<(RequestId, R::Params)>
    where
        R: lsp_types::request::Request + 'static,
        R::Params: DeserializeOwned + 'static,
    {
        let req = self.req.take()?;
        let (id, params) = match req.extract::<R::Params>(R::METHOD) {
            Ok(it) => it,
            Err(req) => {
                self.req = Some(req);
                return None;
            }
        };
        self.incoming_requests.register(id.clone(), (R::METHOD, self.request_received));
        Some((id, params))
    }

    fn finish(&mut self) {
        match self.req.take() {
            None => (),
            Some(req) => {
                log::error!("unknown request: {:?}", req);
                let resp = Response::new_err(
                    req.id,
                    ErrorCode::MethodNotFound as i32,
                    "unknown request".to_string(),
                );
                self.msg_sender.send(resp.into()).unwrap();
            }
        }
    }
}

fn result_to_task<R>(id: RequestId, result: Result<R::Result>) -> Task
where
    R: lsp_types::request::Request + 'static,
    R::Params: DeserializeOwned + 'static,
    R::Result: Serialize + 'static,
{
    let response = match result {
        Ok(resp) => Response::new_ok(id, &resp),
        Err(e) => match e.downcast::<LspError>() {
            Ok(lsp_error) => {
                if lsp_error.code == LspError::UNKNOWN_FILE {
                    // Work-around for https://github.com/rust-analyzer/rust-analyzer/issues/1521
                    Response::new_ok(id, ())
                } else {
                    Response::new_err(id, lsp_error.code, lsp_error.message)
                }
            }
            Err(e) => {
                if is_canceled(&e) {
                    Response::new_err(
                        id,
                        ErrorCode::ContentModified as i32,
                        "content modified".to_string(),
                    )
                } else {
                    Response::new_err(id, ErrorCode::InternalError as i32, e.to_string())
                }
            }
        },
    };
    Task::Respond(response)
}

fn update_file_notifications_on_threadpool(
    pool: &ThreadPool,
    world: GlobalStateSnapshot,
    task_sender: Sender<Task>,
    subscriptions: Vec<FileId>,
) {
    log::trace!("updating notifications for {:?}", subscriptions);
    if world.config.publish_diagnostics {
        pool.execute(move || {
            for file_id in subscriptions {
                match handlers::publish_diagnostics(&world, file_id) {
                    Err(e) => {
                        if !is_canceled(&e) {
                            log::error!("failed to compute diagnostics: {:?}", e);
                        }
                    }
                    Ok(task) => {
                        task_sender.send(Task::Diagnostic(task)).unwrap();
                    }
                }
            }
        })
    }
}

pub fn show_message(
    typ: lsp_types::MessageType,
    message: impl Into<String>,
    sender: &Sender<Message>,
) {
    let message = message.into();
    let params = lsp_types::ShowMessageParams { typ, message };
    let not = notification_new::<lsp_types::notification::ShowMessage>(params);
    sender.send(not.into()).unwrap();
}

fn is_canceled(e: &Box<dyn std::error::Error + Send + Sync>) -> bool {
    e.downcast_ref::<Canceled>().is_some()
}

fn notification_is<N: lsp_types::notification::Notification>(notification: &Notification) -> bool {
    notification.method == N::METHOD
}

fn notification_cast<N>(notification: Notification) -> std::result::Result<N::Params, Notification>
where
    N: lsp_types::notification::Notification,
    N::Params: DeserializeOwned,
{
    notification.extract(N::METHOD)
}

fn notification_new<N>(params: N::Params) -> Notification
where
    N: lsp_types::notification::Notification,
    N::Params: Serialize,
{
    Notification::new(N::METHOD.to_string(), params)
}

#[cfg(test)]
mod tests {
    use lsp_types::{Position, Range, TextDocumentContentChangeEvent};

    use super::*;

    #[test]
    fn test_apply_document_changes() {
        macro_rules! c {
            [$($sl:expr, $sc:expr; $el:expr, $ec:expr => $text:expr),+] => {
                vec![$(TextDocumentContentChangeEvent {
                    range: Some(Range {
                        start: Position { line: $sl, character: $sc },
                        end: Position { line: $el, character: $ec },
                    }),
                    range_length: None,
                    text: String::from($text),
                }),+]
            };
        }

        let mut text = String::new();
        apply_document_changes(&mut text, vec![]);
        assert_eq!(text, "");
        apply_document_changes(
            &mut text,
            vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: String::from("the"),
            }],
        );
        assert_eq!(text, "the");
        apply_document_changes(&mut text, c![0, 3; 0, 3 => " quick"]);
        assert_eq!(text, "the quick");
        apply_document_changes(&mut text, c![0, 0; 0, 4 => "", 0, 5; 0, 5 => " foxes"]);
        assert_eq!(text, "quick foxes");
        apply_document_changes(&mut text, c![0, 11; 0, 11 => "\ndream"]);
        assert_eq!(text, "quick foxes\ndream");
        apply_document_changes(&mut text, c![1, 0; 1, 0 => "have "]);
        assert_eq!(text, "quick foxes\nhave dream");
        apply_document_changes(
            &mut text,
            c![0, 0; 0, 0 => "the ", 1, 4; 1, 4 => " quiet", 1, 16; 1, 16 => "s\n"],
        );
        assert_eq!(text, "the quick foxes\nhave quiet dreams\n");
        apply_document_changes(&mut text, c![0, 15; 0, 15 => "\n", 2, 17; 2, 17 => "\n"]);
        assert_eq!(text, "the quick foxes\n\nhave quiet dreams\n\n");
        apply_document_changes(
            &mut text,
            c![1, 0; 1, 0 => "DREAM", 2, 0; 2, 0 => "they ", 3, 0; 3, 0 => "DON'T THEY?"],
        );
        assert_eq!(text, "the quick foxes\nDREAM\nthey have quiet dreams\nDON'T THEY?\n");
        apply_document_changes(&mut text, c![0, 10; 1, 5 => "", 2, 0; 2, 12 => ""]);
        assert_eq!(text, "the quick \nthey have quiet dreams\n");

        text = String::from("❤️");
        apply_document_changes(&mut text, c![0, 0; 0, 0 => "a"]);
        assert_eq!(text, "a❤️");

        text = String::from("a\nb");
        apply_document_changes(&mut text, c![0, 1; 1, 0 => "\nțc", 0, 1; 1, 1 => "d"]);
        assert_eq!(text, "adcb");

        text = String::from("a\nb");
        apply_document_changes(&mut text, c![0, 1; 1, 0 => "ț\nc", 0, 2; 0, 2 => "c"]);
        assert_eq!(text, "ațc\ncb");
    }
}
