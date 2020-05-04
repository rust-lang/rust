//! The main loop of `rust-analyzer` responsible for dispatching LSP
//! requests/replies and notifications back to the client.

mod handlers;
mod subscriptions;
pub(crate) mod pending_requests;

use std::{
    borrow::Cow,
    env,
    error::Error,
    fmt,
    ops::Range,
    panic,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam_channel::{never, select, unbounded, RecvError, Sender};
use itertools::Itertools;
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::{
    DidChangeTextDocumentParams, NumberOrString, TextDocumentContentChangeEvent, WorkDoneProgress,
    WorkDoneProgressBegin, WorkDoneProgressCreateParams, WorkDoneProgressEnd,
    WorkDoneProgressReport,
};
use ra_flycheck::{url_from_path_with_drive_lowercasing, CheckTask};
use ra_ide::{Canceled, FileId, LibraryData, LineIndex, SourceRootId};
use ra_prof::profile;
use ra_project_model::{PackageRoot, ProjectWorkspace};
use ra_vfs::{VfsFile, VfsTask, Watch};
use relative_path::RelativePathBuf;
use rustc_hash::FxHashSet;
use serde::{de::DeserializeOwned, Serialize};
use threadpool::ThreadPool;

use crate::{
    config::{Config, FilesWatcher},
    conv::{ConvWith, TryConvWith},
    diagnostics::DiagnosticTask,
    main_loop::{
        pending_requests::{PendingRequest, PendingRequests},
        subscriptions::Subscriptions,
    },
    req,
    world::{WorldSnapshot, WorldState},
    Result,
};

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

pub fn main_loop(ws_roots: Vec<PathBuf>, config: Config, connection: Connection) -> Result<()> {
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
    let mut world_state = {
        let workspaces = {
            // FIXME: support dynamic workspace loading.
            let mut visited = FxHashSet::default();
            let project_roots = ws_roots
                .iter()
                .filter_map(|it| ra_project_model::ProjectRoot::discover(it).ok())
                .flatten()
                .filter(|it| visited.insert(it.clone()))
                .collect::<Vec<_>>();

            if project_roots.is_empty() && config.notifications.cargo_toml_not_found {
                show_message(
                        req::MessageType::Error,
                        format!(
                            "rust-analyzer failed to discover workspace, no Cargo.toml found, dirs searched: {}",
                            ws_roots.iter().format_with(", ", |it, f| f(&it.display()))
                        ),
                        &connection.sender,
                    );
            };

            project_roots
                .into_iter()
                .filter_map(|root| {
                    ra_project_model::ProjectWorkspace::load(
                        root,
                        &config.cargo,
                        config.with_sysroot,
                    )
                    .map_err(|err| {
                        log::error!("failed to load workspace: {:#}", err);
                        show_message(
                            req::MessageType::Error,
                            format!("rust-analyzer failed to load workspace: {:#}", err),
                            &connection.sender,
                        );
                    })
                    .ok()
                })
                .collect::<Vec<_>>()
        };

        let globs = config
            .files
            .exclude
            .iter()
            .map(|glob| crate::vfs_glob::Glob::new(glob))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if let FilesWatcher::Client = config.files.watcher {
            let registration_options = req::DidChangeWatchedFilesRegistrationOptions {
                watchers: workspaces
                    .iter()
                    .flat_map(ProjectWorkspace::to_roots)
                    .filter(PackageRoot::is_member)
                    .map(|root| format!("{}/**/*.rs", root.path().display()))
                    .map(|glob_pattern| req::FileSystemWatcher { glob_pattern, kind: None })
                    .collect(),
            };
            let registration = req::Registration {
                id: "file-watcher".to_string(),
                method: "workspace/didChangeWatchedFiles".to_string(),
                register_options: Some(serde_json::to_value(registration_options).unwrap()),
            };
            let params = req::RegistrationParams { registrations: vec![registration] };
            let request =
                request_new::<req::RegisterCapability>(loop_state.next_request_id(), params);
            connection.sender.send(request.into()).unwrap();
        }

        WorldState::new(
            ws_roots,
            workspaces,
            config.lru_capacity,
            &globs,
            Watch(matches!(config.files.watcher, FilesWatcher::Notify)),
            config,
        )
    };

    loop_state.roots_total = world_state.vfs.read().n_roots();
    loop_state.roots_scanned = 0;

    let pool = ThreadPool::default();
    let (task_sender, task_receiver) = unbounded::<Task>();
    let (libdata_sender, libdata_receiver) = unbounded::<LibraryData>();

    log::info!("server initialized, serving requests");
    {
        let task_sender = task_sender;
        let libdata_sender = libdata_sender;
        loop {
            log::trace!("selecting");
            let event = select! {
                recv(&connection.receiver) -> msg => match msg {
                    Ok(msg) => Event::Msg(msg),
                    Err(RecvError) => return Err("client exited without shutdown".into()),
                },
                recv(task_receiver) -> task => Event::Task(task.unwrap()),
                recv(world_state.task_receiver) -> task => match task {
                    Ok(task) => Event::Vfs(task),
                    Err(RecvError) => return Err("vfs died".into()),
                },
                recv(libdata_receiver) -> data => Event::Lib(data.unwrap()),
                recv(world_state.flycheck.as_ref().map_or(&never(), |it| &it.task_recv)) -> task => match task {
                    Ok(task) => Event::CheckWatcher(task),
                    Err(RecvError) => return Err("check watcher died".into()),
                }
            };
            if let Event::Msg(Message::Request(req)) = &event {
                if connection.handle_shutdown(&req)? {
                    break;
                };
            }
            loop_turn(
                &pool,
                &task_sender,
                &libdata_sender,
                &connection,
                &mut world_state,
                &mut loop_state,
                event,
            )?;
        }
    }
    world_state.analysis_host.request_cancellation();
    log::info!("waiting for tasks to finish...");
    task_receiver.into_iter().for_each(|task| {
        on_task(task, &connection.sender, &mut loop_state.pending_requests, &mut world_state)
    });
    libdata_receiver.into_iter().for_each(drop);
    log::info!("...tasks have finished");
    log::info!("joining threadpool...");
    pool.join();
    drop(pool);
    log::info!("...threadpool has finished");

    let vfs = Arc::try_unwrap(world_state.vfs).expect("all snapshots should be dead");
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
    Vfs(VfsTask),
    Lib(LibraryData),
    CheckWatcher(CheckTask),
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let debug_verbose_not = |not: &Notification, f: &mut fmt::Formatter| {
            f.debug_struct("Notification").field("method", &not.method).finish()
        };

        match self {
            Event::Msg(Message::Notification(not)) => {
                if notification_is::<req::DidOpenTextDocument>(not)
                    || notification_is::<req::DidChangeTextDocument>(not)
                {
                    return debug_verbose_not(not, f);
                }
            }
            Event::Task(Task::Notify(not)) => {
                if notification_is::<req::PublishDiagnostics>(not) {
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
            Event::Lib(it) => fmt::Debug::fmt(it, f),
            Event::CheckWatcher(it) => fmt::Debug::fmt(it, f),
        }
    }
}

#[derive(Debug, Default)]
struct LoopState {
    next_request_id: u64,
    pending_responses: FxHashSet<RequestId>,
    pending_requests: PendingRequests,
    subscriptions: Subscriptions,
    // We try not to index more than MAX_IN_FLIGHT_LIBS libraries at the same
    // time to always have a thread ready to react to input.
    in_flight_libraries: usize,
    pending_libraries: Vec<(SourceRootId, Vec<(FileId, RelativePathBuf, Arc<String>)>)>,
    workspace_loaded: bool,
    roots_progress_reported: Option<usize>,
    roots_scanned: usize,
    roots_total: usize,
    configuration_request_id: Option<RequestId>,
}

impl LoopState {
    fn next_request_id(&mut self) -> RequestId {
        self.next_request_id += 1;
        let res: RequestId = self.next_request_id.into();
        let inserted = self.pending_responses.insert(res.clone());
        assert!(inserted);
        res
    }
}

fn loop_turn(
    pool: &ThreadPool,
    task_sender: &Sender<Task>,
    libdata_sender: &Sender<LibraryData>,
    connection: &Connection,
    world_state: &mut WorldState,
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

    match event {
        Event::Task(task) => {
            on_task(task, &connection.sender, &mut loop_state.pending_requests, world_state);
            world_state.maybe_collect_garbage();
        }
        Event::Vfs(task) => {
            world_state.vfs.write().handle_task(task);
        }
        Event::Lib(lib) => {
            world_state.add_lib(lib);
            world_state.maybe_collect_garbage();
            loop_state.in_flight_libraries -= 1;
            loop_state.roots_scanned += 1;
        }
        Event::CheckWatcher(task) => on_check_task(task, world_state, task_sender)?,
        Event::Msg(msg) => match msg {
            Message::Request(req) => on_request(
                world_state,
                &mut loop_state.pending_requests,
                pool,
                task_sender,
                &connection.sender,
                loop_start,
                req,
            )?,
            Message::Notification(not) => {
                on_notification(&connection.sender, world_state, loop_state, not)?;
            }
            Message::Response(resp) => {
                let removed = loop_state.pending_responses.remove(&resp.id);
                if !removed {
                    log::error!("unexpected response: {:?}", resp)
                }

                if Some(&resp.id) == loop_state.configuration_request_id.as_ref() {
                    loop_state.configuration_request_id = None;
                    log::debug!("config update response: '{:?}", resp);
                    let Response { error, result, .. } = resp;

                    match (error, result) {
                        (Some(err), _) => {
                            log::error!("failed to fetch the server settings: {:?}", err)
                        }
                        (None, Some(configs)) => {
                            if let Some(new_config) = configs.get(0) {
                                let mut config = world_state.config.clone();
                                config.update(&new_config);
                                world_state.update_configuration(config);
                            }
                        }
                        (None, None) => {
                            log::error!("received empty server settings response from the client")
                        }
                    }
                }
            }
        },
    };

    let mut state_changed = false;
    if let Some(changes) = world_state.process_changes(&mut loop_state.roots_scanned) {
        state_changed = true;
        loop_state.pending_libraries.extend(changes);
    }

    let max_in_flight_libs = pool.max_count().saturating_sub(2).max(1);
    while loop_state.in_flight_libraries < max_in_flight_libs
        && !loop_state.pending_libraries.is_empty()
    {
        let (root, files) = loop_state.pending_libraries.pop().unwrap();
        loop_state.in_flight_libraries += 1;
        let sender = libdata_sender.clone();
        pool.execute(move || {
            log::info!("indexing {:?} ... ", root);
            let data = LibraryData::prepare(root, files);
            sender.send(data).unwrap();
        });
    }

    let show_progress = !loop_state.workspace_loaded;

    if !loop_state.workspace_loaded
        && loop_state.roots_scanned == loop_state.roots_total
        && loop_state.pending_libraries.is_empty()
        && loop_state.in_flight_libraries == 0
    {
        state_changed = true;
        loop_state.workspace_loaded = true;
        if let Some(flycheck) = &world_state.flycheck {
            flycheck.update();
        }
    }

    if show_progress {
        send_startup_progress(&connection.sender, loop_state);
    }

    if state_changed && loop_state.workspace_loaded {
        update_file_notifications_on_threadpool(
            pool,
            world_state.snapshot(),
            task_sender.clone(),
            loop_state.subscriptions.subscriptions(),
        );
        pool.execute({
            let subs = loop_state.subscriptions.subscriptions();
            let snap = world_state.snapshot();
            move || snap.analysis().prime_caches(subs).unwrap_or_else(|_: Canceled| ())
        });
    }

    let loop_duration = loop_start.elapsed();
    if loop_duration > Duration::from_millis(100) {
        log::error!("overly long loop turn: {:?}", loop_duration);
        if env::var("RA_PROFILE").is_ok() {
            show_message(
                req::MessageType::Error,
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
    pending_requests: &mut PendingRequests,
    state: &mut WorldState,
) {
    match task {
        Task::Respond(response) => {
            if let Some(completed) = pending_requests.finish(&response.id) {
                log::info!("handled req#{} in {:?}", completed.id, completed.duration);
                state.complete_request(completed);
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
    world: &mut WorldState,
    pending_requests: &mut PendingRequests,
    pool: &ThreadPool,
    task_sender: &Sender<Task>,
    msg_sender: &Sender<Message>,
    request_received: Instant,
    req: Request,
) -> Result<()> {
    let mut pool_dispatcher = PoolDispatcher {
        req: Some(req),
        pool,
        world,
        task_sender,
        msg_sender,
        pending_requests,
        request_received,
    };
    pool_dispatcher
        .on_sync::<req::CollectGarbage>(|s, ()| Ok(s.collect_garbage()))?
        .on_sync::<req::JoinLines>(|s, p| handlers::handle_join_lines(s.snapshot(), p))?
        .on_sync::<req::OnEnter>(|s, p| handlers::handle_on_enter(s.snapshot(), p))?
        .on_sync::<req::SelectionRangeRequest>(|s, p| {
            handlers::handle_selection_range(s.snapshot(), p)
        })?
        .on_sync::<req::FindMatchingBrace>(|s, p| {
            handlers::handle_find_matching_brace(s.snapshot(), p)
        })?
        .on::<req::AnalyzerStatus>(handlers::handle_analyzer_status)?
        .on::<req::SyntaxTree>(handlers::handle_syntax_tree)?
        .on::<req::ExpandMacro>(handlers::handle_expand_macro)?
        .on::<req::OnTypeFormatting>(handlers::handle_on_type_formatting)?
        .on::<req::DocumentSymbolRequest>(handlers::handle_document_symbol)?
        .on::<req::WorkspaceSymbol>(handlers::handle_workspace_symbol)?
        .on::<req::GotoDefinition>(handlers::handle_goto_definition)?
        .on::<req::GotoImplementation>(handlers::handle_goto_implementation)?
        .on::<req::GotoTypeDefinition>(handlers::handle_goto_type_definition)?
        .on::<req::ParentModule>(handlers::handle_parent_module)?
        .on::<req::Runnables>(handlers::handle_runnables)?
        .on::<req::Completion>(handlers::handle_completion)?
        .on::<req::CodeActionRequest>(handlers::handle_code_action)?
        .on::<req::CodeLensRequest>(handlers::handle_code_lens)?
        .on::<req::CodeLensResolve>(handlers::handle_code_lens_resolve)?
        .on::<req::FoldingRangeRequest>(handlers::handle_folding_range)?
        .on::<req::SignatureHelpRequest>(handlers::handle_signature_help)?
        .on::<req::HoverRequest>(handlers::handle_hover)?
        .on::<req::PrepareRenameRequest>(handlers::handle_prepare_rename)?
        .on::<req::Rename>(handlers::handle_rename)?
        .on::<req::References>(handlers::handle_references)?
        .on::<req::Formatting>(handlers::handle_formatting)?
        .on::<req::DocumentHighlightRequest>(handlers::handle_document_highlight)?
        .on::<req::InlayHints>(handlers::handle_inlay_hints)?
        .on::<req::CallHierarchyPrepare>(handlers::handle_call_hierarchy_prepare)?
        .on::<req::CallHierarchyIncomingCalls>(handlers::handle_call_hierarchy_incoming)?
        .on::<req::CallHierarchyOutgoingCalls>(handlers::handle_call_hierarchy_outgoing)?
        .on::<req::SemanticTokensRequest>(handlers::handle_semantic_tokens)?
        .on::<req::SemanticTokensRangeRequest>(handlers::handle_semantic_tokens_range)?
        .on::<req::Ssr>(handlers::handle_ssr)?
        .finish();
    Ok(())
}

fn on_notification(
    msg_sender: &Sender<Message>,
    state: &mut WorldState,
    loop_state: &mut LoopState,
    not: Notification,
) -> Result<()> {
    let not = match notification_cast::<req::Cancel>(not) {
        Ok(params) => {
            let id: RequestId = match params.id {
                NumberOrString::Number(id) => id.into(),
                NumberOrString::String(id) => id.into(),
            };
            if loop_state.pending_requests.cancel(&id) {
                let response = Response::new_err(
                    id,
                    ErrorCode::RequestCanceled as i32,
                    "canceled by client".to_string(),
                );
                msg_sender.send(response.into()).unwrap()
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidOpenTextDocument>(not) {
        Ok(params) => {
            let uri = params.text_document.uri;
            let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
            if let Some(file_id) =
                state.vfs.write().add_file_overlay(&path, params.text_document.text)
            {
                loop_state.subscriptions.add_sub(FileId(file_id.0));
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidChangeTextDocument>(not) {
        Ok(params) => {
            let DidChangeTextDocumentParams { text_document, content_changes } = params;
            let world = state.snapshot();
            let file_id = text_document.try_conv_with(&world)?;
            let line_index = world.analysis().file_line_index(file_id)?;
            let uri = text_document.uri;
            let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
            state.vfs.write().change_file_overlay(&path, |old_text| {
                apply_document_changes(old_text, Cow::Borrowed(&line_index), content_changes);
            });
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidSaveTextDocument>(not) {
        Ok(_params) => {
            if let Some(flycheck) = &state.flycheck {
                flycheck.update();
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidCloseTextDocument>(not) {
        Ok(params) => {
            let uri = params.text_document.uri;
            let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
            if let Some(file_id) = state.vfs.write().remove_file_overlay(path.as_path()) {
                loop_state.subscriptions.remove_sub(FileId(file_id.0));
            }
            let params =
                req::PublishDiagnosticsParams { uri, diagnostics: Vec::new(), version: None };
            let not = notification_new::<req::PublishDiagnostics>(params);
            msg_sender.send(not.into()).unwrap();
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidChangeConfiguration>(not) {
        Ok(_) => {
            // As stated in https://github.com/microsoft/language-server-protocol/issues/676,
            // this notification's parameters should be ignored and the actual config queried separately.
            let request_id = loop_state.next_request_id();
            let request = request_new::<req::WorkspaceConfiguration>(
                request_id.clone(),
                req::ConfigurationParams {
                    items: vec![req::ConfigurationItem {
                        scope_uri: None,
                        section: Some("rust-analyzer".to_string()),
                    }],
                },
            );
            msg_sender.send(request.into())?;
            loop_state.configuration_request_id = Some(request_id);

            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidChangeWatchedFiles>(not) {
        Ok(params) => {
            let mut vfs = state.vfs.write();
            for change in params.changes {
                let uri = change.uri;
                let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
                vfs.notify_changed(path)
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
    mut line_index: Cow<'_, LineIndex>,
    content_changes: Vec<TextDocumentContentChangeEvent>,
) {
    // Remove when https://github.com/rust-analyzer/rust-analyzer/issues/4263 is fixed.
    let backup_text = old_text.clone();
    let backup_changes = content_changes.clone();

    // The changes we got must be applied sequentially, but can cross lines so we
    // have to keep our line index updated.
    // Some clients (e.g. Code) sort the ranges in reverse. As an optimization, we
    // remember the last valid line in the index and only rebuild it if needed.
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
                    line_index = Cow::Owned(LineIndex::new(&old_text));
                }
                index_valid = IndexValid::UpToLineExclusive(range.start.line);
                let range = range.conv_with(&line_index);
                let mut text = old_text.to_owned();
                match std::panic::catch_unwind(move || {
                    text.replace_range(Range::<usize>::from(range), &change.text);
                    text
                }) {
                    Ok(t) => *old_text = t,
                    Err(e) => {
                        eprintln!("Bug in incremental text synchronization. Please report the following output on https://github.com/rust-analyzer/rust-analyzer/issues/4263");
                        dbg!(&backup_text);
                        dbg!(&backup_changes);
                        std::panic::resume_unwind(e);
                    }
                }
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
    world_state: &mut WorldState,
    task_sender: &Sender<Task>,
) -> Result<()> {
    match task {
        CheckTask::ClearDiagnostics => {
            task_sender.send(Task::Diagnostic(DiagnosticTask::ClearCheck))?;
        }

        CheckTask::AddDiagnostic { url, diagnostic, fixes } => {
            let path = url.to_file_path().map_err(|()| format!("invalid uri: {}", url))?;
            let file_id = match world_state.vfs.read().path2file(&path) {
                Some(file) => FileId(file.0),
                None => {
                    log::error!("File with cargo diagnostic not found in VFS: {}", path.display());
                    return Ok(());
                }
            };

            task_sender
                .send(Task::Diagnostic(DiagnosticTask::AddCheck(file_id, diagnostic, fixes)))?;
        }

        CheckTask::Status(progress) => {
            let params = req::ProgressParams {
                token: req::ProgressToken::String("rustAnalyzer/cargoWatcher".to_string()),
                value: req::ProgressParamsValue::WorkDone(progress),
            };
            let not = notification_new::<req::Progress>(params);
            task_sender.send(Task::Notify(not)).unwrap();
        }
    };

    Ok(())
}

fn on_diagnostic_task(task: DiagnosticTask, msg_sender: &Sender<Message>, state: &mut WorldState) {
    let subscriptions = state.diagnostics.handle_task(task);

    for file_id in subscriptions {
        let path = state.vfs.read().file2path(VfsFile(file_id.0));
        let uri = match url_from_path_with_drive_lowercasing(&path) {
            Ok(uri) => uri,
            Err(err) => {
                log::error!("Couldn't convert path to url ({}): {}", err, path.display());
                continue;
            }
        };

        let diagnostics = state.diagnostics.diagnostics_for(file_id).cloned().collect();
        let params = req::PublishDiagnosticsParams { uri, diagnostics, version: None };
        let not = notification_new::<req::PublishDiagnostics>(params);
        msg_sender.send(not.into()).unwrap();
    }
}

fn send_startup_progress(sender: &Sender<Message>, loop_state: &mut LoopState) {
    let total: usize = loop_state.roots_total;
    let prev = loop_state.roots_progress_reported;
    let progress = loop_state.roots_scanned;
    loop_state.roots_progress_reported = Some(progress);

    match (prev, loop_state.workspace_loaded) {
        (None, false) => {
            let work_done_progress_create = request_new::<req::WorkDoneProgressCreate>(
                loop_state.next_request_id(),
                WorkDoneProgressCreateParams {
                    token: req::ProgressToken::String("rustAnalyzer/startup".into()),
                },
            );
            sender.send(work_done_progress_create.into()).unwrap();
            send_startup_progress_notif(
                sender,
                WorkDoneProgress::Begin(WorkDoneProgressBegin {
                    title: "rust-analyzer".into(),
                    cancellable: None,
                    message: Some(format!("{}/{} packages", progress, total)),
                    percentage: Some(100.0 * progress as f64 / total as f64),
                }),
            );
        }
        (Some(prev), false) if progress != prev => send_startup_progress_notif(
            sender,
            WorkDoneProgress::Report(WorkDoneProgressReport {
                cancellable: None,
                message: Some(format!("{}/{} packages", progress, total)),
                percentage: Some(100.0 * progress as f64 / total as f64),
            }),
        ),
        (_, true) => send_startup_progress_notif(
            sender,
            WorkDoneProgress::End(WorkDoneProgressEnd {
                message: Some(format!("rust-analyzer loaded, {} packages", progress)),
            }),
        ),
        _ => {}
    }

    fn send_startup_progress_notif(sender: &Sender<Message>, work_done_progress: WorkDoneProgress) {
        let notif = notification_new::<req::Progress>(req::ProgressParams {
            token: req::ProgressToken::String("rustAnalyzer/startup".into()),
            value: req::ProgressParamsValue::WorkDone(work_done_progress),
        });
        sender.send(notif.into()).unwrap();
    }
}

struct PoolDispatcher<'a> {
    req: Option<Request>,
    pool: &'a ThreadPool,
    world: &'a mut WorldState,
    pending_requests: &'a mut PendingRequests,
    msg_sender: &'a Sender<Message>,
    task_sender: &'a Sender<Task>,
    request_received: Instant,
}

impl<'a> PoolDispatcher<'a> {
    /// Dispatches the request onto the current thread
    fn on_sync<R>(
        &mut self,
        f: fn(&mut WorldState, R::Params) -> Result<R::Result>,
    ) -> Result<&mut Self>
    where
        R: req::Request + 'static,
        R::Params: DeserializeOwned + panic::UnwindSafe + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => {
                return Ok(self);
            }
        };
        let world = panic::AssertUnwindSafe(&mut *self.world);
        let task = panic::catch_unwind(move || {
            let result = f(world.0, params);
            result_to_task::<R>(id, result)
        })
        .map_err(|_| format!("sync task {:?} panicked", R::METHOD))?;
        on_task(task, self.msg_sender, self.pending_requests, self.world);
        Ok(self)
    }

    /// Dispatches the request onto thread pool
    fn on<R>(&mut self, f: fn(WorldSnapshot, R::Params) -> Result<R::Result>) -> Result<&mut Self>
    where
        R: req::Request + 'static,
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
            let world = self.world.snapshot();
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
        R: req::Request + 'static,
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
        self.pending_requests.start(PendingRequest {
            id: id.clone(),
            method: R::METHOD.to_string(),
            received: self.request_received,
        });
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
    R: req::Request + 'static,
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
    world: WorldSnapshot,
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

pub fn show_message(typ: req::MessageType, message: impl Into<String>, sender: &Sender<Message>) {
    let message = message.into();
    let params = req::ShowMessageParams { typ, message };
    let not = notification_new::<req::ShowMessage>(params);
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

fn request_new<R>(id: RequestId, params: R::Params) -> Request
where
    R: lsp_types::request::Request,
    R::Params: Serialize,
{
    Request::new(id, R::METHOD.to_string(), params)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use lsp_types::{Position, Range, TextDocumentContentChangeEvent};
    use ra_ide::LineIndex;

    #[test]
    fn apply_document_changes() {
        fn run(text: &mut String, changes: Vec<TextDocumentContentChangeEvent>) {
            let line_index = Cow::Owned(LineIndex::new(&text));
            super::apply_document_changes(text, line_index, changes);
        }

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
        run(&mut text, vec![]);
        assert_eq!(text, "");
        run(
            &mut text,
            vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: String::from("the"),
            }],
        );
        assert_eq!(text, "the");
        run(&mut text, c![0, 3; 0, 3 => " quick"]);
        assert_eq!(text, "the quick");
        run(&mut text, c![0, 0; 0, 4 => "", 0, 5; 0, 5 => " foxes"]);
        assert_eq!(text, "quick foxes");
        run(&mut text, c![0, 11; 0, 11 => "\ndream"]);
        assert_eq!(text, "quick foxes\ndream");
        run(&mut text, c![1, 0; 1, 0 => "have "]);
        assert_eq!(text, "quick foxes\nhave dream");
        run(&mut text, c![0, 0; 0, 0 => "the ", 1, 4; 1, 4 => " quiet", 1, 16; 1, 16 => "s\n"]);
        assert_eq!(text, "the quick foxes\nhave quiet dreams\n");
        run(&mut text, c![0, 15; 0, 15 => "\n", 2, 17; 2, 17 => "\n"]);
        assert_eq!(text, "the quick foxes\n\nhave quiet dreams\n\n");
        run(
            &mut text,
            c![1, 0; 1, 0 => "DREAM", 2, 0; 2, 0 => "they ", 3, 0; 3, 0 => "DON'T THEY?"],
        );
        assert_eq!(text, "the quick foxes\nDREAM\nthey have quiet dreams\nDON'T THEY?\n");
        run(&mut text, c![0, 10; 1, 5 => "", 2, 0; 2, 12 => ""]);
        assert_eq!(text, "the quick \nthey have quiet dreams\n");
    }
}
