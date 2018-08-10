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
extern crate url;
extern crate flexi_logger;
extern crate libeditor;
extern crate libanalysis;

mod io;
mod caps;
mod req;
mod dispatch;
mod handlers;

use std::path::PathBuf;

use threadpool::ThreadPool;
use crossbeam_channel::{bounded, Sender, Receiver};
use flexi_logger::Logger;
use languageserver_types::{TextDocumentItem, VersionedTextDocumentIdentifier, TextDocumentIdentifier};
use libanalysis::{WorldState, World};

use ::{
    io::{Io, RawMsg, RawRequest},
    handlers::{handle_syntax_tree, handle_extend_selection, publish_diagnostics},
};

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

fn main() -> Result<()> {
    Logger::with_env_or_str("m=trace, libanalysis=trace")
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
                if let Some((_params, resp)) = dispatch::expect_request::<req::Initialize>(io, req)? {
                    resp.result(io, req::InitializeResult {
                        capabilities: caps::SERVER_CAPABILITIES
                    })?;
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

type Thunk = Box<for<'a> FnBox<&'a mut Io, Result<()>>>;

fn initialized(io: &mut Io) -> Result<()> {
    {
        let mut world = WorldState::new();
        let mut pool = ThreadPool::new(4);
        let (sender, receiver) = bounded::<Thunk>(16);
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
    sender: Sender<Thunk>,
    receiver: Receiver<Thunk>,
) -> Result<()> {
    info!("server initialized, serving requests");
    loop {
        enum Event {
            Msg(RawMsg),
            Thunk(Thunk),
            ReceiverDead,
        }

        let event = select! {
            recv(io.receiver(), msg) => match msg {
                Some(msg) => Event::Msg(msg),
                None => Event::ReceiverDead,
            },
            recv(receiver, thunk) => Event::Thunk(thunk.unwrap()),
        };

        let msg = match event {
            Event::ReceiverDead => {
                io.cleanup_receiver()?;
                unreachable!();
            }
            Event::Thunk(thunk) => {
                thunk.call_box(io)?;
                continue;
            }
            Event::Msg(msg) => msg,
        };

        match msg {
            RawMsg::Request(req) => {
                let mut req = Some(req);
                handle_request_on_threadpool::<req::SyntaxTree>(
                    &mut req, pool, world, &sender, handle_syntax_tree
                )?;
                handle_request_on_threadpool::<req::ExtendSelection>(
                    &mut req, pool, world, &sender, handle_extend_selection
                )?;
                let mut shutdown = false;
                dispatch::handle_request::<req::Shutdown, _>(&mut req, |(), resp| {
                    resp.result(io, ())?;
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
                    let world = world.snapshot();
                    let sender = sender.clone();
                    let uri = params.text_document.uri;
                    pool.execute(move || {
                        match publish_diagnostics(world, uri) {
                            Err(e) => {
                                error!("failed to compute diagnostics: {:?}", e)
                            }
                            Ok(params) => {
                                sender.send(Box::new(|io: &mut Io| {
                                    dispatch::send_notification::<req::PublishDiagnostics>(io, params)
                                }))
                            }
                        }
                    });
                    Ok(())
                })?;
                dispatch::handle_notification::<req::DidChangeTextDocument, _>(&mut not, |mut params| {
                    let path = params.text_document.file_path()?;
                    let text = params.content_changes.pop()
                        .ok_or_else(|| format_err!("empty changes"))?
                        .text;
                    world.change_overlay(path, Some(text));
                    let world = world.snapshot();
                    let sender = sender.clone();
                    let uri = params.text_document.uri;
                    pool.execute(move || {
                        match publish_diagnostics(world, uri) {
                            Err(e) => {
                                error!("failed to compute diagnostics: {:?}", e)
                            }
                            Ok(params) => {
                                sender.send(Box::new(|io: &mut Io| {
                                    dispatch::send_notification::<req::PublishDiagnostics>(io, params)
                                }))
                            }
                        }
                    });
                    Ok(())
                })?;
                dispatch::handle_notification::<req::DidCloseTextDocument, _>(&mut not, |params| {
                    let path = params.text_document.file_path()?;
                    world.change_overlay(path, None);
                    dispatch::send_notification::<req::PublishDiagnostics>(io, req::PublishDiagnosticsParams {
                        uri: params.text_document.uri,
                        diagnostics: Vec::new(),
                    })?;
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
    sender: &Sender<Thunk>,
    f: fn(World, R::Params) -> Result<R::Result>,
) -> Result<()>
{
    dispatch::handle_request::<R, _>(req, |params, resp| {
        let world = world.snapshot();
        let sender = sender.clone();
        pool.execute(move || {
            let res = f(world, params);
            sender.send(Box::new(|io: &mut Io| resp.response(io, res)))
        });
        Ok(())
    })
}

trait FnBox<A, R>: Send {
    fn call_box(self: Box<Self>, a: A) -> R;
}

impl<A, R, F: FnOnce(A) -> R + Send> FnBox<A, R> for F {
    fn call_box(self: Box<F>, a: A) -> R {
        (*self)(a)
    }
}

trait FilePath {
    fn file_path(&self) -> Result<PathBuf>;
}

impl FilePath for TextDocumentItem {
    fn file_path(&self) -> Result<PathBuf> {
        self.uri.file_path()
    }
}

impl FilePath for VersionedTextDocumentIdentifier {
    fn file_path(&self) -> Result<PathBuf> {
        self.uri.file_path()
    }
}

impl FilePath for TextDocumentIdentifier {
    fn file_path(&self) -> Result<PathBuf> {
        self.uri.file_path()
    }
}

impl FilePath for ::url::Url {
    fn file_path(&self) -> Result<PathBuf> {
        self.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", self))
    }
}
