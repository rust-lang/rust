//! The main loop of `ra_lsp_server` responsible for dispatching LSP requests/replies and
//! notifications back to the client.

mod handlers;
mod subscriptions;
pub(crate) mod pending_requests;

use std::{
    env,
    error::Error,
    fmt, panic,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam_channel::{select, unbounded, RecvError, Sender};
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::{ClientCapabilities, NumberOrString};
use ra_cargo_watch::{CheckOptions, CheckTask};
use ra_ide::{Canceled, FeatureFlags, FileId, LibraryData, SourceRootId};
use ra_prof::profile;
use ra_vfs::{VfsTask, Watch};
use relative_path::RelativePathBuf;
use rustc_hash::FxHashSet;
use serde::{de::DeserializeOwned, Serialize};
use threadpool::ThreadPool;

use crate::{
    main_loop::{
        pending_requests::{PendingRequest, PendingRequests},
        subscriptions::Subscriptions,
    },
    req,
    world::{Options, WorldSnapshot, WorldState},
    Result, ServerConfig,
};

#[derive(Debug)]
pub struct LspError {
    pub code: i32,
    pub message: String,
}

impl LspError {
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

pub fn main_loop(
    ws_roots: Vec<PathBuf>,
    client_caps: ClientCapabilities,
    config: ServerConfig,
    connection: Connection,
) -> Result<()> {
    log::info!("server_config: {:#?}", config);

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
        let feature_flags = {
            let mut ff = FeatureFlags::default();
            for (flag, value) in config.feature_flags {
                if ff.set(flag.as_str(), value).is_err() {
                    log::error!("unknown feature flag: {:?}", flag);
                    show_message(
                        req::MessageType::Error,
                        format!("unknown feature flag: {:?}", flag),
                        &connection.sender,
                    );
                }
            }
            ff
        };
        log::info!("feature_flags: {:#?}", feature_flags);

        // FIXME: support dynamic workspace loading.
        let workspaces = {
            let mut loaded_workspaces = Vec::new();
            for ws_root in &ws_roots {
                let workspace = ra_project_model::ProjectWorkspace::discover_with_sysroot(
                    ws_root.as_path(),
                    config.with_sysroot,
                    &config.cargo_features,
                );
                match workspace {
                    Ok(workspace) => loaded_workspaces.push(workspace),
                    Err(e) => {
                        log::error!("loading workspace failed: {}", e);
                        if let Some(ra_project_model::CargoTomlNotFoundError(_)) = e.downcast_ref()
                        {
                            if !feature_flags.get("notifications.cargo-toml-not-found") {
                                continue;
                            }
                        }
                        show_message(
                            req::MessageType::Error,
                            format!("rust-analyzer failed to load workspace: {}", e),
                            &connection.sender,
                        );
                    }
                }
            }
            loaded_workspaces
        };

        let globs = config
            .exclude_globs
            .iter()
            .map(|glob| ra_vfs_glob::Glob::new(glob))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if config.use_client_watching {
            let registration_options = req::DidChangeWatchedFilesRegistrationOptions {
                watchers: workspaces
                    .iter()
                    .flat_map(|ws| ws.to_roots())
                    .filter(|root| root.is_member())
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

        let options = {
            let text_document_caps = client_caps.text_document.as_ref();
            Options {
                publish_decorations: config.publish_decorations,
                supports_location_link: text_document_caps
                    .and_then(|it| it.definition)
                    .and_then(|it| it.link_support)
                    .unwrap_or(false),
                line_folding_only: text_document_caps
                    .and_then(|it| it.folding_range.as_ref())
                    .and_then(|it| it.line_folding_only)
                    .unwrap_or(false),
                max_inlay_hint_length: config.max_inlay_hint_length,
                cargo_watch: CheckOptions {
                    enable: config.cargo_watch_enable,
                    args: config.cargo_watch_args,
                    command: config.cargo_watch_command,
                    all_targets: config.cargo_watch_all_targets,
                },
            }
        };

        WorldState::new(
            ws_roots,
            workspaces,
            config.lru_capacity,
            &globs,
            Watch(!config.use_client_watching),
            options,
            feature_flags,
        )
    };

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
                    Err(RecvError) => Err("client exited without shutdown")?,
                },
                recv(task_receiver) -> task => Event::Task(task.unwrap()),
                recv(world_state.task_receiver) -> task => match task {
                    Ok(task) => Event::Vfs(task),
                    Err(RecvError) => Err("vfs died")?,
                },
                recv(libdata_receiver) -> data => Event::Lib(data.unwrap()),
                recv(world_state.check_watcher.task_recv) -> task => match task {
                    Ok(task) => Event::CheckWatcher(task),
                    Err(RecvError) => Err("check watcher died")?,
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
                if notification_is::<req::PublishDecorations>(not)
                    || notification_is::<req::PublishDiagnostics>(not)
                {
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
        }
        Event::CheckWatcher(task) => on_check_task(pool, task, world_state, task_sender)?,
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
                on_notification(
                    &connection.sender,
                    world_state,
                    &mut loop_state.pending_requests,
                    &mut loop_state.subscriptions,
                    not,
                )?;
            }
            Message::Response(resp) => {
                let removed = loop_state.pending_responses.remove(&resp.id);
                if !removed {
                    log::error!("unexpected response: {:?}", resp)
                }
            }
        },
    };

    let mut state_changed = false;
    if let Some(changes) = world_state.process_changes() {
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
            let _p = profile(&format!("indexed {:?}", root));
            let data = LibraryData::prepare(root, files);
            sender.send(data).unwrap();
        });
    }

    if !loop_state.workspace_loaded
        && world_state.roots_to_scan == 0
        && loop_state.pending_libraries.is_empty()
        && loop_state.in_flight_libraries == 0
    {
        loop_state.workspace_loaded = true;
        let n_packages: usize = world_state.workspaces.iter().map(|it| it.n_packages()).sum();
        if world_state.feature_flags().get("notifications.workspace-loaded") {
            let msg = format!("workspace loaded, {} rust packages", n_packages);
            show_message(req::MessageType::Info, msg, &connection.sender);
        }
        world_state.check_watcher.update();
    }

    if state_changed {
        update_file_notifications_on_threadpool(
            pool,
            world_state.snapshot(),
            world_state.options.publish_decorations,
            task_sender.clone(),
            loop_state.subscriptions.subscriptions(),
        )
    }

    let loop_duration = loop_start.elapsed();
    if loop_duration > Duration::from_millis(10) {
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
        .on::<req::DecorationsRequest>(handlers::handle_decorations)?
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
        .finish();
    Ok(())
}

fn on_notification(
    msg_sender: &Sender<Message>,
    state: &mut WorldState,
    pending_requests: &mut PendingRequests,
    subs: &mut Subscriptions,
    not: Notification,
) -> Result<()> {
    let not = match notification_cast::<req::Cancel>(not) {
        Ok(params) => {
            let id: RequestId = match params.id {
                NumberOrString::Number(id) => id.into(),
                NumberOrString::String(id) => id.into(),
            };
            if pending_requests.cancel(&id) {
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
                subs.add_sub(FileId(file_id.0));
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidChangeTextDocument>(not) {
        Ok(mut params) => {
            let uri = params.text_document.uri;
            let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
            let text =
                params.content_changes.pop().ok_or_else(|| "empty changes".to_string())?.text;
            state.vfs.write().change_file_overlay(path.as_path(), text);
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidSaveTextDocument>(not) {
        Ok(_params) => {
            state.check_watcher.update();
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match notification_cast::<req::DidCloseTextDocument>(not) {
        Ok(params) => {
            let uri = params.text_document.uri;
            let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
            if let Some(file_id) = state.vfs.write().remove_file_overlay(path.as_path()) {
                subs.remove_sub(FileId(file_id.0));
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
        Ok(_params) => {
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
    log::error!("unhandled notification: {:?}", not);
    Ok(())
}

fn on_check_task(
    pool: &ThreadPool,
    task: CheckTask,
    world_state: &mut WorldState,
    task_sender: &Sender<Task>,
) -> Result<()> {
    let urls = match task {
        CheckTask::ClearDiagnostics => {
            let state = Arc::get_mut(&mut world_state.check_watcher.state)
                .expect("couldn't get check watcher state as mutable");
            state.clear()
        }

        CheckTask::AddDiagnostic(url, diagnostic) => {
            let state = Arc::get_mut(&mut world_state.check_watcher.state)
                .expect("couldn't get check watcher state as mutable");
            state.add_diagnostic_with_fixes(url.clone(), diagnostic);
            vec![url]
        }

        CheckTask::Status(progress) => {
            let params = req::ProgressParams {
                token: req::ProgressToken::String("rustAnalyzer/cargoWatcher".to_string()),
                value: req::ProgressParamsValue::WorkDone(progress),
            };
            let not = notification_new::<req::Progress>(params);
            task_sender.send(Task::Notify(not)).unwrap();
            Vec::new()
        }
    };

    let subscriptions = urls
        .into_iter()
        .map(|url| {
            let path = url.to_file_path().map_err(|()| format!("invalid uri: {}", url))?;
            Ok(world_state.vfs.read().path2file(&path).map(|it| FileId(it.0)))
        })
        .filter_map(|res| res.transpose())
        .collect::<Result<Vec<_>>>()?;

    // We manually send a diagnostic update when the watcher asks
    // us to, to avoid the issue of having to change the file to
    // receive updated diagnostics.
    update_file_notifications_on_threadpool(
        pool,
        world_state.snapshot(),
        false,
        task_sender.clone(),
        subscriptions,
    );

    Ok(())
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
            Ok(lsp_error) => Response::new_err(id, lsp_error.code, lsp_error.message),
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
    publish_decorations: bool,
    task_sender: Sender<Task>,
    subscriptions: Vec<FileId>,
) {
    log::trace!("updating notifications for {:?}", subscriptions);
    let publish_diagnostics = world.feature_flags().get("lsp.diagnostics");
    pool.execute(move || {
        for file_id in subscriptions {
            if publish_diagnostics {
                match handlers::publish_diagnostics(&world, file_id) {
                    Err(e) => {
                        if !is_canceled(&e) {
                            log::error!("failed to compute diagnostics: {:?}", e);
                        }
                    }
                    Ok(params) => {
                        let not = notification_new::<req::PublishDiagnostics>(params);
                        task_sender.send(Task::Notify(not)).unwrap();
                    }
                }
            }
            if publish_decorations {
                match handlers::publish_decorations(&world, file_id) {
                    Err(e) => {
                        if !is_canceled(&e) {
                            log::error!("failed to compute decorations: {:?}", e);
                        }
                    }
                    Ok(params) => {
                        let not = notification_new::<req::PublishDecorations>(params);
                        task_sender.send(Task::Notify(not)).unwrap();
                    }
                }
            }
        }
    });
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
