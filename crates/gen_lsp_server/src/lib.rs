#[macro_use]
extern crate failure;
#[macro_use]
extern crate log;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate crossbeam_channel;
extern crate languageserver_types;

mod msg;
mod stdio;

use crossbeam_channel::{Sender, Receiver};
use languageserver_types::{
    ServerCapabilities, InitializeResult,
    request::{Initialize, Shutdown},
    notification::{Initialized, Exit},
};

pub type Result<T> = ::std::result::Result<T, failure::Error>;
pub use {
    msg::{RawMessage, RawRequest, RawResponse, RawResponseError, RawNotification, ErrorCode},
    stdio::{stdio_transport, Threads},
};

pub fn run_server(
    caps: ServerCapabilities,
    server: impl FnOnce(&mut Receiver<RawMessage>, &mut Sender<RawMessage>) -> Result<()>,
    mut receiver: Receiver<RawMessage>,
    mut sender: Sender<RawMessage>,
) -> Result<()> {
    info!("lsp server initializes");
    initialize(&mut receiver, &mut sender, caps)?;
    info!("lsp server initialized, serving requests");
    server(&mut receiver, &mut sender)?;
    info!("lsp server waiting for exit notification");
    match receiver.recv() {
        Some(RawMessage::Notification(n)) => {
            n.cast::<Exit>().map_err(|n| format_err!(
                "unexpected notification during shutdown: {:?}", n
            ))?
        }
        m => bail!("unexpected message during shutdown: {:?}", m)
    }
    info!("lsp server shutdown complete");
    Ok(())
}

pub fn handle_shutdown(req: RawRequest, sender: &Sender<RawMessage>) -> Option<RawRequest> {
    match req.cast::<Shutdown>() {
        Ok((id, ())) => {
            let resp = RawResponse::ok::<Shutdown>(id, ());
            sender.send(RawMessage::Response(resp));
            None
        }
        Err(req) => Some(req),
    }
}

fn initialize(
    receiver: &mut Receiver<RawMessage>,
    sender: &mut Sender<RawMessage>,
    caps: ServerCapabilities,
) -> Result<()> {
    let id = match receiver.recv() {
        Some(RawMessage::Request(req)) => match req.cast::<Initialize>() {
            Err(req) => bail!("expected initialize request, got {:?}", req),
            Ok(req) => req.0,
        }
        msg =>
            bail!("expected initialize request, got {:?}", msg),
    };
    let resp = RawResponse::ok::<Initialize>(id, InitializeResult { capabilities: caps });
    sender.send(RawMessage::Response(resp));
    match receiver.recv() {
        Some(RawMessage::Notification(n)) => {
            n.cast::<Initialized>().map_err(|_| format_err!(
                "expected initialized notification"
            ))?;
        }
        _ => bail!("expected initialized notification"),
    }
    Ok(())
}
