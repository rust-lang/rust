mod handlers;
mod subscriptions;
pub(crate) mod pending_requests;

use std::{error::Error, fmt, path::PathBuf, sync::Arc, time::Instant};

use crossbeam_channel::{select, unbounded, Receiver, RecvError, Sender};
use gen_lsp_server::{
    handle_shutdown, ErrorCode, RawMessage, RawNotification, RawRequest, RawResponse,
};
use lsp_types::{ClientCapabilities, NumberOrString};
use ra_ide_api::{Canceled, FileId, LibraryData};
use ra_prof::profile;
use ra_vfs::VfsTask;
use serde::{de::DeserializeOwned, Serialize};
use threadpool::ThreadPool;

use crate::{
    main_loop::{
        pending_requests::{PendingRequest, PendingRequests},
        subscriptions::Subscriptions,
    },
    project_model::workspace_loader,
    req,
    world::{Options, WorldSnapshot, WorldState},
    Result, ServerConfig,
};

const THREADPOOL_SIZE: usize = 8;
const MAX_IN_FLIGHT_LIBS: usize = THREADPOOL_SIZE - 3;

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
    msg_receiver: &Receiver<RawMessage>,
    msg_sender: &Sender<RawMessage>,
) -> Result<()> {
    log::debug!("server_config: {:?}", config);
    // FIXME: support dynamic workspace loading.
    let workspaces = {
        let ws_worker = workspace_loader();
        let mut loaded_workspaces = Vec::new();
        for ws_root in &ws_roots {
            ws_worker.sender().send(ws_root.clone()).unwrap();
            match ws_worker.receiver().recv().unwrap() {
                Ok(ws) => loaded_workspaces.push(ws),
                Err(e) => {
                    log::error!("loading workspace failed: {}", e);

                    show_message(
                        req::MessageType::Error,
                        format!("rust-analyzer failed to load workspace: {}", e),
                        msg_sender,
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
    let mut state = WorldState::new(
        ws_roots,
        workspaces,
        config.lru_capacity,
        &globs,
        Options {
            publish_decorations: config.publish_decorations,
            show_workspace_loaded: config.show_workspace_loaded,
            supports_location_link: client_caps
                .text_document
                .and_then(|it| it.definition)
                .and_then(|it| it.link_support)
                .unwrap_or(false),
        },
    );

    let pool = ThreadPool::new(THREADPOOL_SIZE);
    let (task_sender, task_receiver) = unbounded::<Task>();
    let mut pending_requests = PendingRequests::default();

    log::info!("server initialized, serving requests");
    let main_res = main_loop_inner(
        &pool,
        msg_sender,
        msg_receiver,
        task_sender,
        task_receiver.clone(),
        &mut state,
        &mut pending_requests,
    );

    log::info!("waiting for tasks to finish...");
    task_receiver
        .into_iter()
        .for_each(|task| on_task(task, msg_sender, &mut pending_requests, &mut state));
    log::info!("...tasks have finished");
    log::info!("joining threadpool...");
    drop(pool);
    log::info!("...threadpool has finished");

    let vfs = Arc::try_unwrap(state.vfs).expect("all snapshots should be dead");
    drop(vfs);

    main_res
}

#[derive(Debug)]
enum Task {
    Respond(RawResponse),
    Notify(RawNotification),
}

enum Event {
    Msg(RawMessage),
    Task(Task),
    Vfs(VfsTask),
    Lib(LibraryData),
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let debug_verbose_not = |not: &RawNotification, f: &mut fmt::Formatter| {
            f.debug_struct("RawNotification").field("method", &not.method).finish()
        };

        match self {
            Event::Msg(RawMessage::Notification(not)) => {
                if not.is::<req::DidOpenTextDocument>() || not.is::<req::DidChangeTextDocument>() {
                    return debug_verbose_not(not, f);
                }
            }
            Event::Task(Task::Notify(not)) => {
                if not.is::<req::PublishDecorations>() || not.is::<req::PublishDiagnostics>() {
                    return debug_verbose_not(not, f);
                }
            }
            Event::Task(Task::Respond(resp)) => {
                return f
                    .debug_struct("RawResponse")
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
        }
    }
}

fn main_loop_inner(
    pool: &ThreadPool,
    msg_sender: &Sender<RawMessage>,
    msg_receiver: &Receiver<RawMessage>,
    task_sender: Sender<Task>,
    task_receiver: Receiver<Task>,
    state: &mut WorldState,
    pending_requests: &mut PendingRequests,
) -> Result<()> {
    let mut subs = Subscriptions::default();
    // We try not to index more than MAX_IN_FLIGHT_LIBS libraries at the same
    // time to always have a thread ready to react to input.
    let mut in_flight_libraries = 0;
    let mut pending_libraries = Vec::new();
    let mut send_workspace_notification = true;

    let (libdata_sender, libdata_receiver) = unbounded();
    loop {
        log::trace!("selecting");
        let event = select! {
            recv(msg_receiver) -> msg => match msg {
                Ok(msg) => Event::Msg(msg),
                Err(RecvError) => Err("client exited without shutdown")?,
            },
            recv(task_receiver) -> task => Event::Task(task.unwrap()),
            recv(state.vfs.read().task_receiver()) -> task => match task {
                Ok(task) => Event::Vfs(task),
                Err(RecvError) => Err("vfs died")?,
            },
            recv(libdata_receiver) -> data => Event::Lib(data.unwrap())
        };
        let loop_start = Instant::now();

        // NOTE: don't count blocking select! call as a loop-turn time
        let _p = profile("main_loop_inner/loop-turn");
        log::info!("loop turn = {:?}", event);
        let queue_count = pool.queued_count();
        if queue_count > 0 {
            log::info!("queued count = {}", queue_count);
        }

        let mut state_changed = false;
        match event {
            Event::Task(task) => {
                on_task(task, msg_sender, pending_requests, state);
                state.maybe_collect_garbage();
            }
            Event::Vfs(task) => {
                state.vfs.write().handle_task(task);
                state_changed = true;
            }
            Event::Lib(lib) => {
                state.add_lib(lib);
                state.maybe_collect_garbage();
                in_flight_libraries -= 1;
            }
            Event::Msg(msg) => match msg {
                RawMessage::Request(req) => {
                    let req = match handle_shutdown(req, msg_sender) {
                        Some(req) => req,
                        None => return Ok(()),
                    };
                    on_request(
                        state,
                        pending_requests,
                        pool,
                        &task_sender,
                        msg_sender,
                        loop_start,
                        req,
                    )?
                }
                RawMessage::Notification(not) => {
                    on_notification(msg_sender, state, pending_requests, &mut subs, not)?;
                    state_changed = true;
                }
                RawMessage::Response(resp) => log::error!("unexpected response: {:?}", resp),
            },
        };

        pending_libraries.extend(state.process_changes());
        while in_flight_libraries < MAX_IN_FLIGHT_LIBS && !pending_libraries.is_empty() {
            let (root, files) = pending_libraries.pop().unwrap();
            in_flight_libraries += 1;
            let sender = libdata_sender.clone();
            pool.execute(move || {
                log::info!("indexing {:?} ... ", root);
                let _p = profile(&format!("indexed {:?}", root));
                let data = LibraryData::prepare(root, files);
                sender.send(data).unwrap();
            });
        }

        if send_workspace_notification
            && state.roots_to_scan == 0
            && pending_libraries.is_empty()
            && in_flight_libraries == 0
        {
            let n_packages: usize = state.workspaces.iter().map(|it| it.n_packages()).sum();
            if state.options.show_workspace_loaded {
                let msg = format!("workspace loaded, {} rust packages", n_packages);
                show_message(req::MessageType::Info, msg, msg_sender);
            }
            // Only send the notification first time
            send_workspace_notification = false;
        }

        if state_changed {
            update_file_notifications_on_threadpool(
                pool,
                state.snapshot(),
                state.options.publish_decorations,
                task_sender.clone(),
                subs.subscriptions(),
            )
        }
    }
}

fn on_task(
    task: Task,
    msg_sender: &Sender<RawMessage>,
    pending_requests: &mut PendingRequests,
    state: &mut WorldState,
) {
    match task {
        Task::Respond(response) => {
            if let Some(completed) = pending_requests.finish(response.id) {
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
    sender: &Sender<Task>,
    msg_sender: &Sender<RawMessage>,
    request_received: Instant,
    req: RawRequest,
) -> Result<()> {
    let mut pool_dispatcher = PoolDispatcher {
        req: Some(req),
        pool,
        world,
        sender,
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
        .on::<req::ExtendSelection>(handlers::handle_extend_selection)?
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
        .finish();
    Ok(())
}

fn on_notification(
    msg_sender: &Sender<RawMessage>,
    state: &mut WorldState,
    pending_requests: &mut PendingRequests,
    subs: &mut Subscriptions,
    not: RawNotification,
) -> Result<()> {
    let not = match not.cast::<req::Cancel>() {
        Ok(params) => {
            let id = match params.id {
                NumberOrString::Number(id) => id,
                NumberOrString::String(id) => {
                    panic!("string id's not supported: {:?}", id);
                }
            };
            if pending_requests.cancel(id) {
                let response = RawResponse::err(
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
    let not = match not.cast::<req::DidOpenTextDocument>() {
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
    let not = match not.cast::<req::DidChangeTextDocument>() {
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
    let not = match not.cast::<req::DidCloseTextDocument>() {
        Ok(params) => {
            let uri = params.text_document.uri;
            let path = uri.to_file_path().map_err(|()| format!("invalid uri: {}", uri))?;
            if let Some(file_id) = state.vfs.write().remove_file_overlay(path.as_path()) {
                subs.remove_sub(FileId(file_id.0));
            }
            let params = req::PublishDiagnosticsParams { uri, diagnostics: Vec::new() };
            let not = RawNotification::new::<req::PublishDiagnostics>(&params);
            msg_sender.send(not.into()).unwrap();
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match not.cast::<req::DidChangeConfiguration>() {
        Ok(_params) => {
            return Ok(());
        }
        Err(not) => not,
    };
    log::error!("unhandled notification: {:?}", not);
    Ok(())
}

struct PoolDispatcher<'a> {
    req: Option<RawRequest>,
    pool: &'a ThreadPool,
    world: &'a mut WorldState,
    pending_requests: &'a mut PendingRequests,
    msg_sender: &'a Sender<RawMessage>,
    sender: &'a Sender<Task>,
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
        R::Params: DeserializeOwned + Send + 'static,
        R::Result: Serialize + 'static,
    {
        let (id, params) = match self.parse::<R>() {
            Some(it) => it,
            None => {
                return Ok(self);
            }
        };
        let result = f(self.world, params);
        let task = result_to_task::<R>(id, result);
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
            let sender = self.sender.clone();
            move || {
                let result = f(world, params);
                let task = result_to_task::<R>(id, result);
                sender.send(task).unwrap();
            }
        });

        Ok(self)
    }

    fn parse<R>(&mut self) -> Option<(u64, R::Params)>
    where
        R: req::Request + 'static,
        R::Params: DeserializeOwned + Send + 'static,
    {
        let req = self.req.take()?;
        let (id, params) = match req.cast::<R>() {
            Ok(it) => it,
            Err(req) => {
                self.req = Some(req);
                return None;
            }
        };
        self.pending_requests.start(PendingRequest {
            id,
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
                let resp = RawResponse::err(
                    req.id,
                    ErrorCode::MethodNotFound as i32,
                    "unknown request".to_string(),
                );
                self.msg_sender.send(resp.into()).unwrap();
            }
        }
    }
}

fn result_to_task<R>(id: u64, result: Result<R::Result>) -> Task
where
    R: req::Request + 'static,
    R::Params: DeserializeOwned + Send + 'static,
    R::Result: Serialize + 'static,
{
    let response = match result {
        Ok(resp) => RawResponse::ok::<R>(id, &resp),
        Err(e) => match e.downcast::<LspError>() {
            Ok(lsp_error) => RawResponse::err(id, lsp_error.code, lsp_error.message),
            Err(e) => {
                if is_canceled(&e) {
                    // FIXME: When https://github.com/Microsoft/vscode-languageserver-node/issues/457
                    // gets fixed, we can return the proper response.
                    // This works around the issue where "content modified" error would continuously
                    // show an message pop-up in VsCode
                    // RawResponse::err(
                    //     id,
                    //     ErrorCode::ContentModified as i32,
                    //     "content modified".to_string(),
                    // )
                    RawResponse {
                        id,
                        result: Some(serde_json::to_value(&()).unwrap()),
                        error: None,
                    }
                } else {
                    RawResponse::err(id, ErrorCode::InternalError as i32, e.to_string())
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
    sender: Sender<Task>,
    subscriptions: Vec<FileId>,
) {
    pool.execute(move || {
        for file_id in subscriptions {
            match handlers::publish_diagnostics(&world, file_id) {
                Err(e) => {
                    if !is_canceled(&e) {
                        log::error!("failed to compute diagnostics: {:?}", e);
                    }
                }
                Ok(params) => {
                    let not = RawNotification::new::<req::PublishDiagnostics>(&params);
                    sender.send(Task::Notify(not)).unwrap();
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
                        let not = RawNotification::new::<req::PublishDecorations>(&params);
                        sender.send(Task::Notify(not)).unwrap();
                    }
                }
            }
        }
    });
}

fn show_message(typ: req::MessageType, message: impl Into<String>, sender: &Sender<RawMessage>) {
    let message = message.into();
    let params = req::ShowMessageParams { typ, message };
    let not = RawNotification::new::<req::ShowMessage>(&params);
    sender.send(not.into()).unwrap();
}

fn is_canceled(e: &Box<dyn std::error::Error + Send + Sync>) -> bool {
    e.downcast_ref::<Canceled>().is_some()
}
