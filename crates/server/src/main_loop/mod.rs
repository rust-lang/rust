mod handlers;

use std::{
    collections::{HashSet},
};

use threadpool::ThreadPool;
use crossbeam_channel::{Sender, Receiver};
use languageserver_types::Url;
use serde_json::to_value;

use {
    req, dispatch,
    Task, Result,
    io::{Io, RawMsg, RawRequest, RawNotification},
    vfs::FileEvent,
    server_world::{ServerWorldState, ServerWorld},
    main_loop::handlers::{
        handle_syntax_tree,
        handle_extend_selection,
        publish_diagnostics,
        publish_decorations,
        handle_document_symbol,
        handle_code_action,
        handle_execute_command,
        handle_workspace_symbol,
        handle_goto_definition,
        handle_find_matching_brace,
        handle_parent_module,
        handle_join_lines,
        handle_completion,
    },
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

    let mut next_request_id = 0;
    let mut pending_requests: HashSet<u64> = HashSet::new();
    let mut fs_events_receiver = Some(&fs_events_receiver);
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
                    Task::Request(mut request) => {
                        request.id = next_request_id;
                        pending_requests.insert(next_request_id);
                        next_request_id += 1;
                        io.send(RawMsg::Request(request));
                    }
                    Task::Respond(response) =>
                        io.send(RawMsg::Response(response)),
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
            }
            Event::Msg(msg) => {
                match msg {
                    RawMsg::Request(req) => {
                        if !on_request(io, &mut state, pool, &task_sender, req)? {
                            return Ok(());
                        }
                    }
                    RawMsg::Notification(not) => {
                        on_notification(io, &mut state, pool, &task_sender, not)?
                    }
                    RawMsg::Response(resp) => {
                        if !pending_requests.remove(&resp.id) {
                            error!("unexpected response: {:?}", resp)
                        }
                    }
                }
            }
        };
    }
}

fn on_request(
    io: &mut Io,
    world: &mut ServerWorldState,
    pool: &ThreadPool,
    sender: &Sender<Task>,
    req: RawRequest,
) -> Result<bool> {
    let mut req = Some(req);
    handle_request_on_threadpool::<req::SyntaxTree>(
        &mut req, pool, world, sender, handle_syntax_tree,
    )?;
    handle_request_on_threadpool::<req::ExtendSelection>(
        &mut req, pool, world, sender, handle_extend_selection,
    )?;
    handle_request_on_threadpool::<req::FindMatchingBrace>(
        &mut req, pool, world, sender, handle_find_matching_brace,
    )?;
    handle_request_on_threadpool::<req::DocumentSymbolRequest>(
        &mut req, pool, world, sender, handle_document_symbol,
    )?;
    handle_request_on_threadpool::<req::CodeActionRequest>(
        &mut req, pool, world, sender, handle_code_action,
    )?;
    handle_request_on_threadpool::<req::WorkspaceSymbol>(
        &mut req, pool, world, sender, handle_workspace_symbol,
    )?;
    handle_request_on_threadpool::<req::GotoDefinition>(
        &mut req, pool, world, sender, handle_goto_definition,
    )?;
    handle_request_on_threadpool::<req::Completion>(
        &mut req, pool, world, sender, handle_completion,
    )?;
    handle_request_on_threadpool::<req::ParentModule>(
        &mut req, pool, world, sender, handle_parent_module,
    )?;
    handle_request_on_threadpool::<req::JoinLines>(
        &mut req, pool, world, sender, handle_join_lines,
    )?;
    dispatch::handle_request::<req::ExecuteCommand, _>(&mut req, |params, resp| {
        io.send(RawMsg::Response(resp.into_response(Ok(None))?));

        let world = world.snapshot();
        let sender = sender.clone();
        pool.execute(move || {
            let (edit, cursor) = match handle_execute_command(world, params) {
                Ok(res) => res,
                Err(e) => return sender.send(Task::Die(e)),
            };
            match to_value(edit) {
                Err(e) => return sender.send(Task::Die(e.into())),
                Ok(params) => {
                    let request = RawRequest {
                        id: 0,
                        method: <req::ApplyWorkspaceEdit as req::ClientRequest>::METHOD.to_string(),
                        params,
                    };
                    sender.send(Task::Request(request))
                }
            }
            if let Some(cursor) = cursor {
                let request = RawRequest {
                    id: 0,
                    method: <req::MoveCursor as req::ClientRequest>::METHOD.to_string(),
                    params: to_value(cursor).unwrap(),
                };
                sender.send(Task::Request(request))
            }
        });
        Ok(())
    })?;

    let mut shutdown = false;
    dispatch::handle_request::<req::Shutdown, _>(&mut req, |(), resp| {
        let resp = resp.into_response(Ok(()))?;
        io.send(RawMsg::Response(resp));
        shutdown = true;
        Ok(())
    })?;
    if shutdown {
        info!("lifecycle: initiating shutdown");
        return Ok(false);
    }
    if let Some(req) = req {
        error!("unknown method: {:?}", req);
        io.send(RawMsg::Response(dispatch::unknown_method(req.id)?));
    }
    Ok(true)
}

fn on_notification(
    io: &mut Io,
    state: &mut ServerWorldState,
    pool: &ThreadPool,
    sender: &Sender<Task>,
    not: RawNotification,
) -> Result<()> {
    let mut not = Some(not);
    dispatch::handle_notification::<req::DidOpenTextDocument, _>(&mut not, |params| {
        let uri = params.text_document.uri;
        let path = uri.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        state.add_mem_file(path, params.text_document.text);
        update_file_notifications_on_threadpool(
            pool,
            state.snapshot(),
            sender.clone(),
            uri,
        );
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
        update_file_notifications_on_threadpool(
            pool,
            state.snapshot(),
            sender.clone(),
            uri,
        );
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidCloseTextDocument, _>(&mut not, |params| {
        let uri = params.text_document.uri;
        let path = uri.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        state.remove_mem_file(path.as_path())?;
        let not = req::PublishDiagnosticsParams {
            uri,
            diagnostics: Vec::new(),
        };
        let not = dispatch::send_notification::<req::PublishDiagnostics>(not);
        io.send(RawMsg::Notification(not));
        Ok(())
    })?;

    if let Some(not) = not {
        error!("unhandled notification: {:?}", not);
    }
    Ok(())
}

fn handle_request_on_threadpool<R: req::ClientRequest>(
    req: &mut Option<RawRequest>,
    pool: &ThreadPool,
    world: &ServerWorldState,
    sender: &Sender<Task>,
    f: fn(ServerWorld, R::Params) -> Result<R::Result>,
) -> Result<()>
{
    dispatch::handle_request::<R, _>(req, |params, resp| {
        let world = world.snapshot();
        let sender = sender.clone();
        pool.execute(move || {
            let res = f(world, params);
            let task = match resp.into_response(res) {
                Ok(resp) => Task::Respond(resp),
                Err(e) => Task::Die(e),
            };
            sender.send(task);
        });
        Ok(())
    })
}

fn update_file_notifications_on_threadpool(
    pool: &ThreadPool,
    world: ServerWorld,
    sender: Sender<Task>,
    uri: Url,
) {
    pool.execute(move || {
        match publish_diagnostics(world.clone(), uri.clone()) {
            Err(e) => {
                error!("failed to compute diagnostics: {:?}", e)
            }
            Ok(params) => {
                let not = dispatch::send_notification::<req::PublishDiagnostics>(params);
                sender.send(Task::Notify(not));
            }
        }
        match publish_decorations(world, uri) {
            Err(e) => {
                error!("failed to compute decorations: {:?}", e)
            }
            Ok(params) => {
                let not = dispatch::send_notification::<req::PublishDecorations>(params);
                sender.send(Task::Notify(not))
            }
        }
    });
}
