mod handlers;
mod subscriptions;

use std::path::PathBuf;

use crossbeam_channel::{unbounded, select, Receiver, Sender};
use gen_lsp_server::{
    handle_shutdown, ErrorCode, RawMessage, RawNotification, RawRequest, RawResponse,
};
use languageserver_types::NumberOrString;
use ra_analysis::{Canceled, FileId, LibraryData};
use rayon;
use thread_worker::Worker;
use threadpool::ThreadPool;
use rustc_hash::FxHashSet;
use serde::{de::DeserializeOwned, Serialize};
use failure::{format_err, bail};
use failure_derive::Fail;

use crate::{
    main_loop::subscriptions::Subscriptions,
    project_model::{workspace_loader, CargoWorkspace},
    req,
    server_world::{ServerWorld, ServerWorldState},
    vfs::{self, FileEvent},
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

pub fn main_loop(
    internal_mode: bool,
    root: PathBuf,
    publish_decorations: bool,
    msg_receiver: &Receiver<RawMessage>,
    msg_sender: &Sender<RawMessage>,
) -> Result<()> {
    let pool = ThreadPool::new(8);
    let (task_sender, task_receiver) = unbounded::<Task>();
    let (fs_worker, fs_watcher) = vfs::roots_loader();
    let (ws_worker, ws_watcher) = workspace_loader();

    log::info!("server initialized, serving requests");
    let mut state = ServerWorldState::default();

    let mut pending_requests = FxHashSet::default();
    let mut subs = Subscriptions::new();
    let main_res = main_loop_inner(
        internal_mode,
        publish_decorations,
        root,
        &pool,
        msg_sender,
        msg_receiver,
        task_sender,
        task_receiver.clone(),
        fs_worker,
        ws_worker,
        &mut state,
        &mut pending_requests,
        &mut subs,
    );

    log::info!("waiting for tasks to finish...");
    task_receiver.for_each(|task| on_task(task, msg_sender, &mut pending_requests));
    log::info!("...tasks have finished");
    log::info!("joining threadpool...");
    drop(pool);
    log::info!("...threadpool has finished");

    let fs_res = fs_watcher.stop();
    let ws_res = ws_watcher.stop();

    main_res?;
    fs_res.map_err(|_| format_err!("fs watcher died"))?;
    ws_res.map_err(|_| format_err!("ws watcher died"))?;

    Ok(())
}

fn main_loop_inner(
    internal_mode: bool,
    publish_decorations: bool,
    ws_root: PathBuf,
    pool: &ThreadPool,
    msg_sender: &Sender<RawMessage>,
    msg_receiver: &Receiver<RawMessage>,
    task_sender: Sender<Task>,
    task_receiver: Receiver<Task>,
    fs_worker: Worker<PathBuf, (PathBuf, Vec<FileEvent>)>,
    ws_worker: Worker<PathBuf, Result<CargoWorkspace>>,
    state: &mut ServerWorldState,
    pending_requests: &mut FxHashSet<u64>,
    subs: &mut Subscriptions,
) -> Result<()> {
    let (libdata_sender, libdata_receiver) = unbounded();
    ws_worker.send(ws_root.clone());
    fs_worker.send(ws_root.clone());
    loop {
        #[derive(Debug)]
        enum Event {
            Msg(RawMessage),
            Task(Task),
            Fs(PathBuf, Vec<FileEvent>),
            Ws(Result<CargoWorkspace>),
            Lib(LibraryData),
        }
        log::trace!("selecting");
        let event = select! {
            recv(msg_receiver, msg) => match msg {
                Some(msg) => Event::Msg(msg),
                None => bail!("client exited without shutdown"),
            },
            recv(task_receiver, task) => Event::Task(task.unwrap()),
            recv(fs_worker.out, events) => match events {
                None => bail!("roots watcher died"),
                Some((pb, events)) => Event::Fs(pb, events),
            }
            recv(ws_worker.out, ws) => match ws {
                None => bail!("workspace watcher died"),
                Some(ws) => Event::Ws(ws),
            }
            recv(libdata_receiver, data) => Event::Lib(data.unwrap())
        };
        let mut state_changed = false;
        match event {
            Event::Task(task) => on_task(task, msg_sender, pending_requests),
            Event::Fs(root, events) => {
                log::info!("fs change, {}, {} events", root.display(), events.len());
                if root == ws_root {
                    state.apply_fs_changes(events);
                } else {
                    let (files, resolver) = state.events_to_files(events);
                    let sender = libdata_sender.clone();
                    pool.execute(move || {
                        let start = ::std::time::Instant::now();
                        log::info!("indexing {} ... ", root.display());
                        let data = LibraryData::prepare(files, resolver);
                        log::info!("indexed {:?} {}", start.elapsed(), root.display());
                        sender.send(data);
                    });
                }
                state_changed = true;
            }
            Event::Ws(ws) => match ws {
                Ok(ws) => {
                    let workspaces = vec![ws];
                    feedback(internal_mode, "workspace loaded", msg_sender);
                    for ws in workspaces.iter() {
                        // Add each library as constant input. If library is
                        // within the workspace, don't treat it as a library.
                        //
                        // HACK: If source roots are nested, pick the outer one.

                        let mut roots = ws
                            .packages()
                            .filter(|pkg| !pkg.is_member(ws))
                            .filter_map(|pkg| {
                                let root = pkg.root(ws).to_path_buf();
                                if root.starts_with(&ws_root) {
                                    None
                                } else {
                                    Some(root)
                                }
                            })
                            .collect::<Vec<_>>();
                        roots.sort_by_key(|it| it.as_os_str().len());
                        let unique = roots
                            .iter()
                            .enumerate()
                            .filter(|&(idx, long)| {
                                !roots[..idx].iter().any(|short| long.starts_with(short))
                            })
                            .map(|(_idx, root)| root);

                        for root in unique {
                            log::debug!("sending root, {}", root.display());
                            fs_worker.send(root.to_owned());
                        }
                    }
                    state.set_workspaces(workspaces);
                    state_changed = true;
                }
                Err(e) => log::warn!("loading workspace failed: {}", e),
            },
            Event::Lib(lib) => {
                feedback(internal_mode, "library loaded", msg_sender);
                state.add_lib(lib);
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
                            msg_sender.send(RawMessage::Response(resp))
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

        if state_changed {
            update_file_notifications_on_threadpool(
                pool,
                state.snapshot(),
                publish_decorations,
                task_sender.clone(),
                subs.subscriptions(),
            )
        }
    }
}

fn on_task(task: Task, msg_sender: &Sender<RawMessage>, pending_requests: &mut FxHashSet<u64>) {
    match task {
        Task::Respond(response) => {
            if pending_requests.remove(&response.id) {
                msg_sender.send(RawMessage::Response(response))
            }
        }
        Task::Notify(n) => msg_sender.send(RawMessage::Notification(n)),
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
        .on::<req::FoldingRangeRequest>(handlers::handle_folding_range)?
        .on::<req::SignatureHelpRequest>(handlers::handle_signature_help)?
        .on::<req::HoverRequest>(handlers::handle_hover)?
        .on::<req::PrepareRenameRequest>(handlers::handle_prepare_rename)?
        .on::<req::Rename>(handlers::handle_rename)?
        .on::<req::References>(handlers::handle_references)?
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
                    ErrorCode::RequestCancelled as i32,
                    "canceled by client".to_string(),
                );
                msg_sender.send(RawMessage::Response(response))
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
            let file_id = state.add_mem_file(path, params.text_document.text);
            subs.add_sub(file_id);
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
            state.change_mem_file(path.as_path(), text)?;
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
            let file_id = state.remove_mem_file(path.as_path())?;
            subs.remove_sub(file_id);
            let params = req::PublishDiagnosticsParams {
                uri,
                diagnostics: Vec::new(),
            };
            let not = RawNotification::new::<req::PublishDiagnostics>(&params);
            msg_sender.send(RawMessage::Notification(not));
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
    fn on<'b, R>(
        &'b mut self,
        f: fn(ServerWorld, R::Params) -> Result<R::Result>,
    ) -> Result<&'b mut Self>
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
                                        ErrorCode::RequestCancelled as i32,
                                        e.to_string(),
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
                    sender.send(task);
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
                    sender.send(Task::Notify(not));
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
                        sender.send(Task::Notify(not))
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
    sender.send(RawMessage::Notification(not));
}

fn is_canceled(e: &failure::Error) -> bool {
    e.downcast_ref::<Canceled>().is_some()
}
