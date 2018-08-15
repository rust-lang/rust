mod handlers;

use std::{
    collections::{HashSet, HashMap},
};

use threadpool::ThreadPool;
use crossbeam_channel::{Sender, Receiver};
use languageserver_types::Url;
use libanalysis::{World, WorldState, FileId};
use serde_json::to_value;

use {
    req, dispatch,
    Task, Result, PathMap,
    io::{Io, RawMsg, RawRequest, RawNotification},
    vfs::{FileEvent, FileEventKind},
    conv::TryConvWith,
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
    },
};

pub(super) fn main_loop(
    io: &mut Io,
    world: &mut WorldState,
    pool: &mut ThreadPool,
    task_sender: Sender<Task>,
    task_receiver: Receiver<Task>,
    fs_events_receiver: Receiver<Vec<FileEvent>>,
) -> Result<()> {
    info!("server initialized, serving requests");
    let mut next_request_id = 0;
    let mut pending_requests: HashSet<u64> = HashSet::new();
    let mut path_map = PathMap::new();
    let mut mem_map: HashMap<FileId, Option<String>> = HashMap::new();
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
                let changes = events.into_iter()
                    .map(|event| {
                        let text = match event.kind {
                            FileEventKind::Add(text) => Some(text),
                            FileEventKind::Remove => None,
                        };
                        (event.path, text)
                    })
                    .map(|(path, text)| {
                        (path_map.get_or_insert(path), text)
                    })
                    .filter_map(|(id, text)| {
                        if mem_map.contains_key(&id) {
                            mem_map.insert(id, text);
                            None
                        } else {
                            Some((id, text))
                        }
                    });

                world.change_files(changes);
            }
            Event::Msg(msg) => {
                match msg {
                    RawMsg::Request(req) => {
                        if !on_request(io, world, &path_map, pool, &task_sender, req)? {
                            return Ok(());
                        }
                    }
                    RawMsg::Notification(not) => {
                        on_notification(io, world, &mut path_map, pool, &task_sender, not, &mut mem_map)?
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
    world: &WorldState,
    path_map: &PathMap,
    pool: &ThreadPool,
    sender: &Sender<Task>,
    req: RawRequest,
) -> Result<bool> {
    let mut req = Some(req);
    handle_request_on_threadpool::<req::SyntaxTree>(
        &mut req, pool, path_map, world, sender, handle_syntax_tree,
    )?;
    handle_request_on_threadpool::<req::ExtendSelection>(
        &mut req, pool, path_map, world, sender, handle_extend_selection,
    )?;
    handle_request_on_threadpool::<req::DocumentSymbolRequest>(
        &mut req, pool, path_map, world, sender, handle_document_symbol,
    )?;
    handle_request_on_threadpool::<req::CodeActionRequest>(
        &mut req, pool, path_map, world, sender, handle_code_action,
    )?;
    handle_request_on_threadpool::<req::WorkspaceSymbol>(
        &mut req, pool, path_map, world, sender, handle_workspace_symbol,
    )?;
    handle_request_on_threadpool::<req::GotoDefinition>(
        &mut req, pool, path_map, world, sender, handle_goto_definition,
    )?;
    dispatch::handle_request::<req::ExecuteCommand, _>(&mut req, |params, resp| {
        io.send(RawMsg::Response(resp.into_response(Ok(None))?));

        let world = world.snapshot();
        let path_map = path_map.clone();
        let sender = sender.clone();
        pool.execute(move || {
            let task = match handle_execute_command(world, path_map, params) {
                Ok(req) => match to_value(req) {
                    Err(e) => Task::Die(e.into()),
                    Ok(params) => {
                        let request = RawRequest {
                            id: 0,
                            method: <req::ApplyWorkspaceEdit as req::ClientRequest>::METHOD.to_string(),
                            params,
                        };
                        Task::Request(request)
                    }
                },
                Err(e) => Task::Die(e),
            };
            sender.send(task)
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
    world: &mut WorldState,
    path_map: &mut PathMap,
    pool: &ThreadPool,
    sender: &Sender<Task>,
    not: RawNotification,
    mem_map: &mut HashMap<FileId, Option<String>>,
) -> Result<()> {
    let mut not = Some(not);
    dispatch::handle_notification::<req::DidOpenTextDocument, _>(&mut not, |params| {
        let uri = params.text_document.uri;
        let path = uri.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", uri))?;
        let file_id = path_map.get_or_insert(path);
        mem_map.insert(file_id, None);
        world.change_file(file_id, Some(params.text_document.text));
        update_file_notifications_on_threadpool(
            pool, world.snapshot(), path_map.clone(), sender.clone(), uri,
        );
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidChangeTextDocument, _>(&mut not, |mut params| {
        let file_id = params.text_document.try_conv_with(path_map)?;
        let text = params.content_changes.pop()
            .ok_or_else(|| format_err!("empty changes"))?
            .text;
        world.change_file(file_id, Some(text));
        update_file_notifications_on_threadpool(
            pool, world.snapshot(), path_map.clone(), sender.clone(), params.text_document.uri,
        );
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidCloseTextDocument, _>(&mut not, |params| {
        let file_id = params.text_document.try_conv_with(path_map)?;
        let text = match mem_map.remove(&file_id) {
            Some(text) => text,
            None => bail!("unmatched close notification"),
        };
        world.change_file(file_id, text);
        let not = req::PublishDiagnosticsParams {
            uri: params.text_document.uri,
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
    path_map: &PathMap,
    world: &WorldState,
    sender: &Sender<Task>,
    f: fn(World, PathMap, R::Params) -> Result<R::Result>,
) -> Result<()>
{
    dispatch::handle_request::<R, _>(req, |params, resp| {
        let world = world.snapshot();
        let path_map = path_map.clone();
        let sender = sender.clone();
        pool.execute(move || {
            let res = f(world, path_map, params);
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
    world: World,
    path_map: PathMap,
    sender: Sender<Task>,
    uri: Url,
) {
    pool.execute(move || {
        match publish_diagnostics(world.clone(), path_map.clone(), uri.clone()) {
            Err(e) => {
                error!("failed to compute diagnostics: {:?}", e)
            }
            Ok(params) => {
                let not = dispatch::send_notification::<req::PublishDiagnostics>(params);
                sender.send(Task::Notify(not));
            }
        }
        match publish_decorations(world, path_map.clone(), uri) {
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
