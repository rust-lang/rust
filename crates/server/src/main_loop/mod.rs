mod handlers;
mod subscriptions;

use std::{
    collections::{HashMap},
};

use threadpool::ThreadPool;
use crossbeam_channel::{Sender, Receiver};
use languageserver_types::{NumberOrString};
use libanalysis::{FileId, JobHandle, JobToken};

use {
    req, dispatch,
    Task, Result,
    io::{Io, RawMsg, RawRequest, RawNotification},
    vfs::FileEvent,
    server_world::{ServerWorldState, ServerWorld},
    main_loop::subscriptions::{Subscriptions},
};

pub(super) fn main_loop(
    io: &mut Io,
    pool: &mut ThreadPool,
    task_sender: Sender<Task>,
    task_receiver: Receiver<Task>,
    fs_events_receiver: Receiver<Vec<FileEvent>>,
) -> Result<()> {
    info!("server initialized, serving requests");
    let mut state = ServerWorldState::new();

    let mut pending_requests: HashMap<u64, JobHandle> = HashMap::new();
    let mut fs_events_receiver = Some(&fs_events_receiver);
    let mut subs = Subscriptions::new();
    loop {
        enum Event {
            Msg(RawMsg),
            Task(Task),
            Fs(Vec<FileEvent>),
            ReceiverDead,
            FsWatcherDead,
        }
        let event = select! {
            recv(io.receiver(), msg) => match msg {
                Some(msg) => Event::Msg(msg),
                None => Event::ReceiverDead,
            },
            recv(task_receiver, task) => Event::Task(task.unwrap()),
            recv(fs_events_receiver, events) => match events {
                Some(events) => Event::Fs(events),
                None => Event::FsWatcherDead,
            }
        };
        let mut state_changed = false;
        match event {
            Event::ReceiverDead => {
                io.cleanup_receiver()?;
                unreachable!();
            }
            Event::FsWatcherDead => {
                fs_events_receiver = None;
            }
            Event::Task(task) => {
                match task {
                    Task::Respond(response) => {
                        if let Some(handle) = pending_requests.remove(&response.id) {
                            assert!(handle.has_completed());
                        }
                        io.send(RawMsg::Response(response))
                    }
                    Task::Notify(n) =>
                        io.send(RawMsg::Notification(n)),
                    Task::Die(error) =>
                        return Err(error),
                }
                continue;
            }
            Event::Fs(events) => {
                trace!("fs change, {} events", events.len());
                state.apply_fs_changes(events);
                state_changed = true;
            }
            Event::Msg(msg) => {
                match msg {
                    RawMsg::Request(req) => {
                        if !on_request(io, &mut state, &mut pending_requests, pool, &task_sender, req)? {
                            return Ok(());
                        }
                    }
                    RawMsg::Notification(not) => {
                        on_notification(io, &mut state, &mut pending_requests, &mut subs, not)?;
                        state_changed = true;
                    }
                    RawMsg::Response(resp) => {
                        error!("unexpected response: {:?}", resp)
                    }
                }
            }
        };

        if state_changed {
            update_file_notifications_on_threadpool(
                pool,
                state.snapshot(),
                task_sender.clone(),
                subs.subscriptions(),
            )
        }
    }
}

fn on_request(
    io: &mut Io,
    world: &mut ServerWorldState,
    pending_requests: &mut HashMap<u64, JobHandle>,
    pool: &ThreadPool,
    sender: &Sender<Task>,
    req: RawRequest,
) -> Result<bool> {
    let mut pool_dispatcher = PoolDispatcher {
        req: Some(req),
        res: None,
        pool, world, sender
    };
    let req = pool_dispatcher
        .on::<req::SyntaxTree>(handlers::handle_syntax_tree)?
        .on::<req::ExtendSelection>(handlers::handle_extend_selection)?
        .on::<req::FindMatchingBrace>(handlers::handle_find_matching_brace)?
        .on::<req::JoinLines>(handlers::handle_join_lines)?
        .on::<req::OnTypeFormatting>(handlers::handle_on_type_formatting)?
        .on::<req::DocumentSymbolRequest>(handlers::handle_document_symbol)?
        .on::<req::WorkspaceSymbol>(handlers::handle_workspace_symbol)?
        .on::<req::GotoDefinition>(handlers::handle_goto_definition)?
        .on::<req::ParentModule>(handlers::handle_parent_module)?
        .on::<req::Runnables>(handlers::handle_runnables)?
        .on::<req::DecorationsRequest>(handlers::handle_decorations)?
        .on::<req::Completion>(handlers::handle_completion)?
        .on::<req::CodeActionRequest>(handlers::handle_code_action)?
        .finish();
    match req {
        Ok((id, handle)) => {
            let inserted = pending_requests.insert(id, handle).is_none();
            assert!(inserted, "duplicate request: {}", id);
        },
        Err(req) => {
            let req = dispatch::handle_request::<req::Shutdown, _>(req, |(), resp| {
                let resp = resp.into_response(Ok(()))?;
                io.send(RawMsg::Response(resp));
                Ok(())
            })?;
            match req {
                Ok(_id) => {
                    info!("lifecycle: initiating shutdown");
                    return Ok(false);
                }
                Err(req) => {
                    error!("unknown method: {:?}", req);
                    io.send(RawMsg::Response(dispatch::unknown_method(req.id)?));
                }
            }
        }
    }
    Ok(true)
}

fn on_notification(
    io: &mut Io,
    state: &mut ServerWorldState,
    pending_requests: &mut HashMap<u64, JobHandle>,
    subs: &mut Subscriptions,
    not: RawNotification,
) -> Result<()> {
    let mut not = Some(not);
    dispatch::handle_notification::<req::Cancel, _>(&mut not, |params| {
        let id = match params.id {
            NumberOrString::Number(id) => id,
            NumberOrString::String(id) => {
                panic!("string id's not supported: {:?}", id);
            }
        };
        if let Some(handle) = pending_requests.remove(&id) {
            handle.cancel();
        }
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidOpenTextDocument, _>(&mut not, |params| {
        let uri = params.text_document.uri;
        let path = uri.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        let file_id = state.add_mem_file(path, params.text_document.text);
        subs.add_sub(file_id);
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidChangeTextDocument, _>(&mut not, |mut params| {
        let uri = params.text_document.uri;
        let path = uri.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        let text = params.content_changes.pop()
            .ok_or_else(|| format_err!("empty changes"))?
            .text;
        state.change_mem_file(path.as_path(), text)?;
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidCloseTextDocument, _>(&mut not, |params| {
        let uri = params.text_document.uri;
        let path = uri.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        let file_id = state.remove_mem_file(path.as_path())?;
        subs.remove_sub(file_id);
        let not = req::PublishDiagnosticsParams { uri, diagnostics: Vec::new() };
        let not = dispatch::send_notification::<req::PublishDiagnostics>(not);
        io.send(RawMsg::Notification(not));
        Ok(())
    })?;

    if let Some(not) = not {
        error!("unhandled notification: {:?}", not);
    }
    Ok(())
}

struct PoolDispatcher<'a> {
    req: Option<RawRequest>,
    res: Option<(u64, JobHandle)>,
    pool: &'a ThreadPool,
    world: &'a ServerWorldState,
    sender: &'a Sender<Task>,
}

impl<'a> PoolDispatcher<'a> {
    fn on<'b, R: req::ClientRequest>(
        &'b mut self,
        f: fn(ServerWorld, R::Params, JobToken) -> Result<R::Result>
    ) -> Result<&'b mut Self> {
        let req = match self.req.take() {
            None => return Ok(self),
            Some(req) => req,
        };
        let world = self.world;
        let sender = self.sender;
        let pool = self.pool;
        let (handle, token) = JobHandle::new();
        let req = dispatch::handle_request::<R, _>(req, |params, resp| {
            let world = world.snapshot();
            let sender = sender.clone();
            pool.execute(move || {
                let res = f(world, params, token);
                let task = match resp.into_response(res) {
                    Ok(resp) => Task::Respond(resp),
                    Err(e) => Task::Die(e),
                };
                sender.send(task);
            });
            Ok(())
        })?;
        match req {
            Ok(id) => self.res = Some((id, handle)),
            Err(req) => self.req = Some(req),
        }
        Ok(self)
    }

    fn finish(&mut self) -> ::std::result::Result<(u64, JobHandle), RawRequest> {
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
    sender: Sender<Task>,
    subscriptions: Vec<FileId>,
) {
    pool.execute(move || {
        for file_id in subscriptions {
            match handlers::publish_diagnostics(world.clone(), file_id) {
                Err(e) => {
                    error!("failed to compute diagnostics: {:?}", e)
                }
                Ok(params) => {
                    let not = dispatch::send_notification::<req::PublishDiagnostics>(params);
                    sender.send(Task::Notify(not));
                }
            }
            match handlers::publish_decorations(world.clone(), file_id) {
                Err(e) => {
                    error!("failed to compute decorations: {:?}", e)
                }
                Ok(params) => {
                    let not = dispatch::send_notification::<req::PublishDecorations>(params);
                    sender.send(Task::Notify(not))
                }
            }
        }
    });
}
