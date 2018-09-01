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
    request::{Initialize},
    notification::{Initialized, Exit},
};

pub type Result<T> = ::std::result::Result<T, failure::Error>;
pub use {
    msg::{RawMessage, RawRequest, RawResponse, RawResponseError, RawNotification},
    stdio::{stdio_transport, Threads},
};

pub type LspServer = fn(&mut Receiver<RawMessage>, &mut Sender<RawMessage>) -> Result<()>;

pub fn run_server(
    caps: ServerCapabilities,
    server: LspServer,
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
    let resp = RawResponse::ok(id, InitializeResult { capabilities: caps });
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
