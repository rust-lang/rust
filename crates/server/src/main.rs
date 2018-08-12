#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate languageserver_types;
extern crate drop_bomb;
#[macro_use]
extern crate crossbeam_channel;
extern crate threadpool;
#[macro_use]
extern crate log;
extern crate url_serde;
extern crate flexi_logger;
extern crate libeditor;
extern crate libanalysis;
extern crate libsyntax2;

mod io;
mod caps;
mod req;
mod dispatch;
mod handlers;
mod util;
mod conv;

use threadpool::ThreadPool;
use crossbeam_channel::{bounded, Sender, Receiver};
use flexi_logger::Logger;
use languageserver_types::Url;
use libanalysis::{WorldState, World};

use ::{
    io::{Io, RawMsg, RawRequest, RawResponse, RawNotification},
    handlers::{handle_syntax_tree, handle_extend_selection, publish_diagnostics, publish_decorations,
               handle_document_symbol, handle_code_action},
    util::FilePath,
};

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

fn main() -> Result<()> {
    Logger::with_env()
        .log_to_file()
        .directory("log")
        .start()?;
    info!("lifecycle: server started");
    match ::std::panic::catch_unwind(|| main_inner()) {
        Ok(res) => {
            info!("lifecycle: terminating process with {:?}", res);
            res
        }
        Err(_) => {
            error!("server panicked");
            bail!("server panicked")
        }
    }
}

fn main_inner() -> Result<()> {
    let mut io = Io::from_stdio();
    let res = initialize(&mut io);
    info!("shutting down IO...");
    let io_res = io.stop();
    info!("... IO is down");
    match (res, io_res) {
        (Ok(()), Ok(())) => Ok(()),
        (res, Ok(())) => res,
        (Ok(()), io_res) => io_res,
        (res, Err(io_err)) => {
            error!("shutdown error: {:?}", io_err);
            res
        }
    }
}

fn initialize(io: &mut Io) -> Result<()> {
    loop {
        match io.recv()? {
            RawMsg::Request(req) => {
                let mut req = Some(req);
                dispatch::handle_request::<req::Initialize, _>(&mut req, |_params, resp| {
                    let res = req::InitializeResult { capabilities: caps::SERVER_CAPABILITIES };
                    let resp = resp.into_response(Ok(res))?;
                    io.send(RawMsg::Response(resp));
                    Ok(())
                })?;
                match req {
                    None => {
                        match io.recv()? {
                            RawMsg::Notification(n) => {
                                if n.method != "initialized" {
                                    bail!("expected initialized notification");
                                }
                            }
                            _ => {
                                bail!("expected initialized notification");
                            }
                        }
                        return initialized(io);
                    }
                    Some(req) => {
                        bail!("expected initialize request, got {:?}", req)
                    }
                }
            }
            RawMsg::Notification(n) => {
                bail!("expected initialize request, got {:?}", n)
            }
            RawMsg::Response(res) => {
                bail!("expected initialize request, got {:?}", res)
            }
        }
    }
}

enum Task {
    Respond(RawResponse),
    Notify(RawNotification),
    Die(::failure::Error),
}

fn initialized(io: &mut Io) -> Result<()> {
    {
        let mut world = WorldState::new();
        let mut pool = ThreadPool::new(4);
        let (sender, receiver) = bounded::<Task>(16);
        info!("lifecycle: handshake finished, server ready to serve requests");
        let res = main_loop(io, &mut world, &mut pool, sender, receiver.clone());
        info!("waiting for background jobs to finish...");
        receiver.for_each(drop);
        pool.join();
        info!("...background jobs have finished");
        res
    }?;

    match io.recv()? {
        RawMsg::Notification(n) => {
            if n.method == "exit" {
                info!("lifecycle: shutdown complete");
                return Ok(());
            }
            bail!("unexpected notification during shutdown: {:?}", n)
        }
        m => {
            bail!("unexpected message during shutdown: {:?}", m)
        }
    }
}

fn main_loop(
    io: &mut Io,
    world: &mut WorldState,
    pool: &mut ThreadPool,
    sender: Sender<Task>,
    receiver: Receiver<Task>,
) -> Result<()> {
    info!("server initialized, serving requests");
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

        let msg = match event {
            Event::ReceiverDead => {
                io.cleanup_receiver()?;
                unreachable!();
            }
            Event::Task(task) => {
                match task {
                    Task::Respond(response) =>
                        io.send(RawMsg::Response(response)),
                    Task::Notify(n) =>
                        io.send(RawMsg::Notification(n)),
                    Task::Die(error) =>
                        return Err(error),
                }
                continue;
            }
            Event::Msg(msg) => msg,
        };

        match msg {
            RawMsg::Request(req) => {
                let mut req = Some(req);
                handle_request_on_threadpool::<req::SyntaxTree>(
                    &mut req, pool, world, &sender, handle_syntax_tree,
                )?;
                handle_request_on_threadpool::<req::ExtendSelection>(
                    &mut req, pool, world, &sender, handle_extend_selection,
                )?;
                handle_request_on_threadpool::<req::DocumentSymbolRequest>(
                    &mut req, pool, world, &sender, handle_document_symbol,
                )?;
                handle_request_on_threadpool::<req::CodeActionRequest>(
                    &mut req, pool, world, &sender, handle_code_action,
                )?;

                let mut shutdown = false;
                dispatch::handle_request::<req::Shutdown, _>(&mut req, |(), resp| {
                    let resp = resp.into_response(Ok(()))?;
                    io.send(RawMsg::Response(resp));
                    shutdown = true;
                    Ok(())
                })?;
                if shutdown {
                    info!("lifecycle: initiating shutdown");
                    drop(sender);
                    return Ok(());
                }
                if let Some(req) = req {
                    error!("unknown method: {:?}", req);
                    dispatch::unknown_method(io, req)?;
                }
            }
            RawMsg::Notification(not) => {
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
                    error!("unhandled notification: {:?}", not)
                }
            }
            msg => {
                eprintln!("msg = {:?}", msg);
            }
        }
    }
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
