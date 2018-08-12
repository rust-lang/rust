mod handlers;

use std::collections::HashSet;

use threadpool::ThreadPool;
use crossbeam_channel::{Sender, Receiver};
use languageserver_types::Url;
use libanalysis::{World, WorldState};
use serde_json::to_value;

use {
    req, dispatch,
    Task, Result,
    io::{Io, RawMsg, RawRequest, RawNotification},
    util::FilePath,
    main_loop::handlers::{
        handle_syntax_tree,
        handle_extend_selection,
        publish_diagnostics,
        publish_decorations,
        handle_document_symbol,
        handle_code_action,
        handle_execute_command,
    },
};

pub(super) fn main_loop(
    io: &mut Io,
    world: &mut WorldState,
    pool: &mut ThreadPool,
    sender: Sender<Task>,
    receiver: Receiver<Task>,
) -> Result<()> {
    info!("server initialized, serving requests");
    let mut next_request_id = 0;
    let mut pending_requests: HashSet<u64> = HashSet::new();
    loop {
        enum Event {
            Msg(RawMsg),
            Task(Task),
            ReceiverDead,
        }
        let event = select! {
            recv(io.receiver(), msg) => match msg {
                Some(msg) => Event::Msg(msg),
                None => Event::ReceiverDead,
            },
            recv(receiver, task) => Event::Task(task.unwrap()),
        };

        match event {
            Event::ReceiverDead => {
                io.cleanup_receiver()?;
                unreachable!();
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
            Event::Msg(msg) => {
                match msg {
                    RawMsg::Request(req) => {
                        if !on_request(io, world, pool, &sender, req)? {
                            return Ok(());
                        }
                    }
                    RawMsg::Notification(not) => {
                        on_notification(io, world, pool, &sender, not)?
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
    handle_request_on_threadpool::<req::DocumentSymbolRequest>(
        &mut req, pool, world, sender, handle_document_symbol,
    )?;
    handle_request_on_threadpool::<req::CodeActionRequest>(
        &mut req, pool, world, sender, handle_code_action,
    )?;
    dispatch::handle_request::<req::ExecuteCommand, _>(&mut req, |params, resp| {
        io.send(RawMsg::Response(resp.into_response(Ok(None))?));

        let world = world.snapshot();
        let sender = sender.clone();
        pool.execute(move || {
            let task = match handle_execute_command(world, params) {
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
    pool: &ThreadPool,
    sender: &Sender<Task>,
    not: RawNotification,
) -> Result<()> {
    let mut not = Some(not);
    dispatch::handle_notification::<req::DidOpenTextDocument, _>(&mut not, |params| {
        let path = params.text_document.file_path()?;
        world.change_overlay(path, Some(params.text_document.text));
        update_file_notifications_on_threadpool(
            pool, world.snapshot(), sender.clone(), params.text_document.uri,
        );
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidChangeTextDocument, _>(&mut not, |mut params| {
        let path = params.text_document.file_path()?;
        let text = params.content_changes.pop()
            .ok_or_else(|| format_err!("empty changes"))?
            .text;
        world.change_overlay(path, Some(text));
        update_file_notifications_on_threadpool(
            pool, world.snapshot(), sender.clone(), params.text_document.uri,
        );
        Ok(())
    })?;
    dispatch::handle_notification::<req::DidCloseTextDocument, _>(&mut not, |params| {
        let path = params.text_document.file_path()?;
        world.change_overlay(path, None);
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
    world: &WorldState,
    sender: &Sender<Task>,
    f: fn(World, R::Params) -> Result<R::Result>,
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
    world: World,
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
