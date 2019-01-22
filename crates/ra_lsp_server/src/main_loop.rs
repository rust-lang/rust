mod handlers;
mod subscriptions;

use std::{fmt, path::PathBuf, sync::Arc};

use crossbeam_channel::{select, unbounded, Receiver, RecvError, Sender};
use failure::{bail, format_err};
use failure_derive::Fail;
use gen_lsp_server::{
    handle_shutdown, ErrorCode, RawMessage, RawNotification, RawRequest, RawResponse,
};
use lsp_types::NumberOrString;
use ra_ide_api::{Canceled, FileId, LibraryData};
use ra_vfs::VfsTask;
use rustc_hash::FxHashSet;
use serde::{de::DeserializeOwned, Serialize};
use threadpool::ThreadPool;

use crate::{
    main_loop::subscriptions::Subscriptions,
    project_model::workspace_loader,
    req,
    server_world::{ServerWorld, ServerWorldState},
    Result,
};

#[derive(Debug, Fail)]
#[fail(
    display = "Language Server request failed with {}. ({})",
    code, message
)]
pub struct LspError {
    pub code: i32,
    pub message: String,
}

impl LspError {
    pub fn new(code: i32, message: String) -> LspError {
        LspError { code, message }
    }
}

#[derive(Debug)]
enum Task {
    Respond(RawResponse),
    Notify(RawNotification),
}

const THREADPOOL_SIZE: usize = 8;

pub fn main_loop(
    internal_mode: bool,
    ws_root: PathBuf,
    supports_decorations: bool,
    msg_receiver: &Receiver<RawMessage>,
    msg_sender: &Sender<RawMessage>,
) -> Result<()> {
    let pool = ThreadPool::new(THREADPOOL_SIZE);
    let (task_sender, task_receiver) = unbounded::<Task>();
    let (ws_worker, ws_watcher) = workspace_loader();

    ws_worker.send(ws_root.clone()).unwrap();
    // FIXME: support dynamic workspace loading.
    let workspaces = match ws_worker.recv().unwrap() {
        Ok(ws) => vec![ws],
        Err(e) => {
            log::error!("loading workspace failed: {}", e);
            Vec::new()
        }
    };
    ws_worker.shutdown();
    ws_watcher
        .shutdown()
        .map_err(|_| format_err!("ws watcher died"))?;
    let mut state = ServerWorldState::new(ws_root.clone(), workspaces);

    log::info!("server initialized, serving requests");

    let mut pending_requests = FxHashSet::default();
    let mut subs = Subscriptions::new();
    let main_res = main_loop_inner(
        internal_mode,
        supports_decorations,
        &pool,
        msg_sender,
        msg_receiver,
        task_sender,
        task_receiver.clone(),
        &mut state,
        &mut pending_requests,
        &mut subs,
    );

    log::info!("waiting for tasks to finish...");
    task_receiver
        .into_iter()
        .for_each(|task| on_task(task, msg_sender, &mut pending_requests));
    log::info!("...tasks have finished");
    log::info!("joining threadpool...");
    drop(pool);
    log::info!("...threadpool has finished");

    let vfs = Arc::try_unwrap(state.vfs).expect("all snapshots should be dead");
    let vfs_res = vfs.into_inner().shutdown();

    main_res?;
    vfs_res.map_err(|_| format_err!("fs watcher died"))?;

    Ok(())
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
            f.debug_struct("RawNotification")
                .field("method", &not.method)
                .finish()
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
    internal_mode: bool,
    supports_decorations: bool,
    pool: &ThreadPool,
    msg_sender: &Sender<RawMessage>,
    msg_receiver: &Receiver<RawMessage>,
    task_sender: Sender<Task>,
    task_receiver: Receiver<Task>,
    state: &mut ServerWorldState,
    pending_requests: &mut FxHashSet<u64>,
    subs: &mut Subscriptions,
) -> Result<()> {
    // We try not to index more than THREADPOOL_SIZE - 3 libraries at the same
    // time to always have a thread ready to react to input.
    let mut in_flight_libraries = 0;
    let mut pending_libraries = Vec::new();

    let (libdata_sender, libdata_receiver) = unbounded();
    loop {
        log::trace!("selecting");
        let event = select! {
            recv(msg_receiver) -> msg => match msg {
                Ok(msg) => Event::Msg(msg),
                Err(RecvError) => bail!("client exited without shutdown"),
            },
            recv(task_receiver) -> task => Event::Task(task.unwrap()),
            recv(state.vfs.read().task_receiver()) -> task => match task {
                Ok(task) => Event::Vfs(task),
                Err(RecvError) => bail!("vfs died"),
            },
            recv(libdata_receiver) -> data => Event::Lib(data.unwrap())
        };
        log::info!("loop_turn = {:?}", event);
        let start = std::time::Instant::now();
        let mut state_changed = false;
        match event {
            Event::Task(task) => on_task(task, msg_sender, pending_requests),
            Event::Vfs(task) => {
                state.vfs.write().handle_task(task);
                state_changed = true;
            }
            Event::Lib(lib) => {
                feedback(internal_mode, "library loaded", msg_sender);
                state.add_lib(lib);
                in_flight_libraries -= 1;
            }
            Event::Msg(msg) => match msg {
                RawMessage::Request(req) => {
                    let req = match handle_shutdown(req, msg_sender) {
                        Some(req) => req,
                        None => return Ok(()),
                    };
                    match on_request(state, pending_requests, pool, &task_sender, req)? {
                        None => (),
                        Some(req) => {
                            log::error!("unknown request: {:?}", req);
                            let resp = RawResponse::err(
                                req.id,
                                ErrorCode::MethodNotFound as i32,
                                "unknown request".to_string(),
                            );
                            msg_sender.send(RawMessage::Response(resp)).unwrap()
                        }
                    }
                }
                RawMessage::Notification(not) => {
                    on_notification(msg_sender, state, pending_requests, subs, not)?;
                    state_changed = true;
                }
                RawMessage::Response(resp) => log::error!("unexpected response: {:?}", resp),
            },
        };

        pending_libraries.extend(state.process_changes());
        while in_flight_libraries < THREADPOOL_SIZE - 3 && !pending_libraries.is_empty() {
            let (root, files) = pending_libraries.pop().unwrap();
            in_flight_libraries += 1;
            let sender = libdata_sender.clone();
            pool.execute(move || {
                let start = ::std::time::Instant::now();
                log::info!("indexing {:?} ... ", root);
                let data = LibraryData::prepare(root, files);
                log::info!("indexed {:?} {:?}", start.elapsed(), root);
                sender.send(data).unwrap();
            });
        }

        if state.roots_to_scan == 0 && pending_libraries.is_empty() && in_flight_libraries == 0 {
            feedback(internal_mode, "workspace loaded", msg_sender);
        }

        if state_changed {
            update_file_notifications_on_threadpool(
                pool,
                state.snapshot(),
                supports_decorations,
                task_sender.clone(),
                subs.subscriptions(),
            )
        }
        log::info!("loop_turn = {:?}", start.elapsed());
    }
}

fn on_task(task: Task, msg_sender: &Sender<RawMessage>, pending_requests: &mut FxHashSet<u64>) {
    match task {
        Task::Respond(response) => {
            if pending_requests.remove(&response.id) {
                msg_sender.send(RawMessage::Response(response)).unwrap();
            }
        }
        Task::Notify(n) => {
            msg_sender.send(RawMessage::Notification(n)).unwrap();
        }
    }
}

fn on_request(
    world: &mut ServerWorldState,
    pending_requests: &mut FxHashSet<u64>,
    pool: &ThreadPool,
    sender: &Sender<Task>,
    req: RawRequest,
) -> Result<Option<RawRequest>> {
    let mut pool_dispatcher = PoolDispatcher {
        req: Some(req),
        res: None,
        pool,
        world,
        sender,
    };
    let req = pool_dispatcher
        .on::<req::AnalyzerStatus>(handlers::handle_analyzer_status)?
        .on::<req::SyntaxTree>(handlers::handle_syntax_tree)?
        .on::<req::ExtendSelection>(handlers::handle_extend_selection)?
        .on::<req::FindMatchingBrace>(handlers::handle_find_matching_brace)?
        .on::<req::JoinLines>(handlers::handle_join_lines)?
        .on::<req::OnEnter>(handlers::handle_on_enter)?
        .on::<req::OnTypeFormatting>(handlers::handle_on_type_formatting)?
        .on::<req::DocumentSymbolRequest>(handlers::handle_document_symbol)?
        .on::<req::WorkspaceSymbol>(handlers::handle_workspace_symbol)?
        .on::<req::GotoDefinition>(handlers::handle_goto_definition)?
        .on::<req::ParentModule>(handlers::handle_parent_module)?
        .on::<req::Runnables>(handlers::handle_runnables)?
        .on::<req::DecorationsRequest>(handlers::handle_decorations)?
        .on::<req::Completion>(handlers::handle_completion)?
        .on::<req::CodeActionRequest>(handlers::handle_code_action)?
        .on::<req::CodeLensRequest>(handlers::handle_code_lens)?
        .on::<req::FoldingRangeRequest>(handlers::handle_folding_range)?
        .on::<req::SignatureHelpRequest>(handlers::handle_signature_help)?
        .on::<req::HoverRequest>(handlers::handle_hover)?
        .on::<req::PrepareRenameRequest>(handlers::handle_prepare_rename)?
        .on::<req::Rename>(handlers::handle_rename)?
        .on::<req::References>(handlers::handle_references)?
        .on::<req::Formatting>(handlers::handle_formatting)?
        .on::<req::DocumentHighlightRequest>(handlers::handle_document_highlight)?
        .finish();
    match req {
        Ok(id) => {
            let inserted = pending_requests.insert(id);
            assert!(inserted, "duplicate request: {}", id);
            Ok(None)
        }
        Err(req) => Ok(Some(req)),
    }
}

fn on_notification(
    msg_sender: &Sender<RawMessage>,
    state: &mut ServerWorldState,
    pending_requests: &mut FxHashSet<u64>,
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
            if pending_requests.remove(&id) {
                let response = RawResponse::err(
                    id,
                    ErrorCode::RequestCanceled as i32,
                    "canceled by client".to_string(),
                );
                msg_sender.send(RawMessage::Response(response)).unwrap()
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match not.cast::<req::DidOpenTextDocument>() {
        Ok(params) => {
            let uri = params.text_document.uri;
            let path = uri
                .to_file_path()
                .map_err(|()| format_err!("invalid uri: {}", uri))?;
            if let Some(file_id) = state
                .vfs
                .write()
                .add_file_overlay(&path, params.text_document.text)
            {
                subs.add_sub(FileId(file_id.0.into()));
            }
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match not.cast::<req::DidChangeTextDocument>() {
        Ok(mut params) => {
            let uri = params.text_document.uri;
            let path = uri
                .to_file_path()
                .map_err(|()| format_err!("invalid uri: {}", uri))?;
            let text = params
                .content_changes
                .pop()
                .ok_or_else(|| format_err!("empty changes"))?
                .text;
            state.vfs.write().change_file_overlay(path.as_path(), text);
            return Ok(());
        }
        Err(not) => not,
    };
    let not = match not.cast::<req::DidCloseTextDocument>() {
        Ok(params) => {
            let uri = params.text_document.uri;
            let path = uri
                .to_file_path()
                .map_err(|()| format_err!("invalid uri: {}", uri))?;
            if let Some(file_id) = state.vfs.write().remove_file_overlay(path.as_path()) {
                subs.remove_sub(FileId(file_id.0.into()));
            }
            let params = req::PublishDiagnosticsParams {
                uri,
                diagnostics: Vec::new(),
            };
            let not = RawNotification::new::<req::PublishDiagnostics>(&params);
            msg_sender.send(RawMessage::Notification(not)).unwrap();
            return Ok(());
        }
        Err(not) => not,
    };
    log::error!("unhandled notification: {:?}", not);
    Ok(())
}

struct PoolDispatcher<'a> {
    req: Option<RawRequest>,
    res: Option<u64>,
    pool: &'a ThreadPool,
    world: &'a ServerWorldState,
    sender: &'a Sender<Task>,
}

impl<'a> PoolDispatcher<'a> {
    fn on<R>(&mut self, f: fn(ServerWorld, R::Params) -> Result<R::Result>) -> Result<&mut Self>
    where
        R: req::Request,
        R::Params: DeserializeOwned + Send + 'static,
        R::Result: Serialize + 'static,
    {
        let req = match self.req.take() {
            None => return Ok(self),
            Some(req) => req,
        };
        match req.cast::<R>() {
            Ok((id, params)) => {
                let world = self.world.snapshot();
                let sender = self.sender.clone();
                self.pool.execute(move || {
                    let resp = match f(world, params) {
                        Ok(resp) => RawResponse::ok::<R>(id, &resp),
                        Err(e) => match e.downcast::<LspError>() {
                            Ok(lsp_error) => {
                                RawResponse::err(id, lsp_error.code, lsp_error.message)
                            }
                            Err(e) => {
                                if is_canceled(&e) {
                                    RawResponse::err(
                                        id,
                                        ErrorCode::ContentModified as i32,
                                        "content modified".to_string(),
                                    )
                                } else {
                                    RawResponse::err(
                                        id,
                                        ErrorCode::InternalError as i32,
                                        format!("{}\n{}", e, e.backtrace()),
                                    )
                                }
                            }
                        },
                    };
                    let task = Task::Respond(resp);
                    sender.send(task).unwrap();
                });
                self.res = Some(id);
            }
            Err(req) => self.req = Some(req),
        }
        Ok(self)
    }

    fn finish(&mut self) -> ::std::result::Result<u64, RawRequest> {
        match (self.res.take(), self.req.take()) {
            (Some(res), None) => Ok(res),
            (None, Some(req)) => Err(req),
            _ => unreachable!(),
        }
    }
}

fn update_file_notifications_on_threadpool(
    pool: &ThreadPool,
    world: ServerWorld,
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

fn feedback(intrnal_mode: bool, msg: &str, sender: &Sender<RawMessage>) {
    if !intrnal_mode {
        return;
    }
    let not = RawNotification::new::<req::InternalFeedback>(&msg.to_string());
    sender.send(RawMessage::Notification(not)).unwrap();
}

fn is_canceled(e: &failure::Error) -> bool {
    e.downcast_ref::<Canceled>().is_some()
}
