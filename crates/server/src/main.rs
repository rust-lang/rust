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
extern crate walkdir;
extern crate libeditor;
extern crate libanalysis;
extern crate libsyntax2;

mod io;
mod caps;
mod req;
mod dispatch;
mod util;
mod conv;
mod main_loop;
mod vfs;

use std::path::PathBuf;

use threadpool::ThreadPool;
use crossbeam_channel::bounded;
use flexi_logger::{Logger, Duplicate};
use libanalysis::WorldState;

use ::{
    io::{Io, RawMsg, RawResponse, RawRequest, RawNotification}
};

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

fn main() -> Result<()> {
    Logger::with_env()
        .duplicate_to_stderr(Duplicate::All)
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
    match io.recv()? {
        RawMsg::Notification(n) =>
            bail!("expected initialize request, got {:?}", n),
        RawMsg::Response(res) =>
            bail!("expected initialize request, got {:?}", res),

        RawMsg::Request(req) => {
            let mut req = Some(req);
            dispatch::handle_request::<req::Initialize, _>(&mut req, |_params, resp| {
                let res = req::InitializeResult { capabilities: caps::server_capabilities() };
                let resp = resp.into_response(Ok(res))?;
                io.send(RawMsg::Response(resp));
                Ok(())
            })?;
            if let Some(req) = req {
                bail!("expected initialize request, got {:?}", req)
            }
            match io.recv()? {
                RawMsg::Notification(n) => {
                    if n.method != "initialized" {
                        bail!("expected initialized notification");
                    }
                }
                _ => bail!("expected initialized notification"),
            }
        }
    }
    initialized(io)
}

enum Task {
    Respond(RawResponse),
    Request(RawRequest),
    Notify(RawNotification),
    Die(::failure::Error),
}

fn initialized(io: &mut Io) -> Result<()> {
    {
        let mut world = WorldState::new();
        let mut pool = ThreadPool::new(4);
        let (task_sender, task_receiver) = bounded::<Task>(16);
        let (fs_events_receiver, watcher) = vfs::watch(vec![
            PathBuf::from("./")
        ]);
        info!("lifecycle: handshake finished, server ready to serve requests");
        let res = main_loop::main_loop(
            io,
            &mut world,
            &mut pool,
            task_sender,
            task_receiver.clone(),
            fs_events_receiver,
        );

        info!("waiting for background jobs to finish...");
        task_receiver.for_each(drop);
        pool.join();
        info!("...background jobs have finished");

        info!("waiting for file watcher to finish...");
        watcher.stop()?;
        info!("...file watcher has finished");

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
