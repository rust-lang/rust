//! A language server scaffold, exposing synchroneous crossbeam-channel based API.
//! This crate handles protocol handshaking and parsing messages, while you
//! control the message dispatch loop yourself.
//!
//! Run with `RUST_LOG=sync_lsp_server=debug` to see all the messages.
//!
//! ```no_run
//! extern crate gen_lsp_server;
//! extern crate languageserver_types;
//! extern crate failure;
//! extern crate crossbeam_channel;
//!
//! use crossbeam_channel::{Sender, Receiver};
//! use languageserver_types::{ServerCapabilities, InitializeParams, request::{GotoDefinition, GotoDefinitionResponse}};
//! use gen_lsp_server::{run_server, stdio_transport, handle_shutdown, RawMessage, RawResponse};
//!
//! fn main() -> Result<(), failure::Error> {
//!     let (receiver, sender, io_threads) = stdio_transport();
//!     gen_lsp_server::run_server(
//!         ServerCapabilities::default(),
//!         receiver,
//!         sender,
//!         main_loop,
//!     )?;
//!     io_threads.join()?;
//!     Ok(())
//! }
//!
//! fn main_loop(
//!     _params: InitializeParams,
//!     receiver: &Receiver<RawMessage>,
//!     sender: &Sender<RawMessage>,
//! ) -> Result<(), failure::Error> {
//!     for msg in receiver {
//!         match msg {
//!             RawMessage::Request(req) => {
//!                 let req = match handle_shutdown(req, sender) {
//!                     None => return Ok(()),
//!                     Some(req) => req,
//!                 };
//!                 let req = match req.cast::<GotoDefinition>() {
//!                     Ok((id, _params)) => {
//!                         let resp = RawResponse::ok::<GotoDefinition>(
//!                             id,
//!                             &Some(GotoDefinitionResponse::Array(Vec::new())),
//!                         );
//!                         sender.send(RawMessage::Response(resp));
//!                         continue;
//!                     },
//!                     Err(req) => req,
//!                 };
//!                 // ...
//!             }
//!             RawMessage::Response(_resp) => (),
//!             RawMessage::Notification(_not) => (),
//!         }
//!     }
//!     Ok(())
//! }
//! ```

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

use crossbeam_channel::{Receiver, Sender};
use languageserver_types::{
    notification::{Exit, Initialized},
    request::{Initialize, Shutdown},
    InitializeParams, InitializeResult, ServerCapabilities,
};

pub type Result<T> = ::std::result::Result<T, failure::Error>;
pub use {
    msg::{ErrorCode, RawMessage, RawNotification, RawRequest, RawResponse, RawResponseError},
    stdio::{stdio_transport, Threads},
};

/// Main entry point: runs the server from initialization to shutdown.
/// To attach server to standard input/output streams, use `stdio_transport`
/// function to create corresponding `sender` and `receiver` pair.
///
///`server` should use `handle_shutdown` function to handle the `Shutdown`
/// request.
pub fn run_server(
    caps: ServerCapabilities,
    receiver: Receiver<RawMessage>,
    sender: Sender<RawMessage>,
    server: impl FnOnce(InitializeParams, &Receiver<RawMessage>, &Sender<RawMessage>) -> Result<()>,
) -> Result<()> {
    info!("lsp server initializes");
    let params = initialize(&receiver, &sender, caps)?;
    info!("lsp server initialized, serving requests");
    server(params, &receiver, &sender)?;
    info!("lsp server waiting for exit notification");
    match receiver.recv() {
        Some(RawMessage::Notification(n)) => n
            .cast::<Exit>()
            .map_err(|n| format_err!("unexpected notification during shutdown: {:?}", n))?,
        m => bail!("unexpected message during shutdown: {:?}", m),
    }
    info!("lsp server shutdown complete");
    Ok(())
}

/// if `req` is `Shutdown`, respond to it and return `None`, otherwise return `Some(req)`
pub fn handle_shutdown(req: RawRequest, sender: &Sender<RawMessage>) -> Option<RawRequest> {
    match req.cast::<Shutdown>() {
        Ok((id, ())) => {
            let resp = RawResponse::ok::<Shutdown>(id, &());
            sender.send(RawMessage::Response(resp));
            None
        }
        Err(req) => Some(req),
    }
}

fn initialize(
    receiver: &Receiver<RawMessage>,
    sender: &Sender<RawMessage>,
    caps: ServerCapabilities,
) -> Result<InitializeParams> {
    let (id, params) = match receiver.recv() {
        Some(RawMessage::Request(req)) => match req.cast::<Initialize>() {
            Err(req) => bail!("expected initialize request, got {:?}", req),
            Ok(req) => req,
        },
        msg => bail!("expected initialize request, got {:?}", msg),
    };
    let resp = RawResponse::ok::<Initialize>(id, &InitializeResult { capabilities: caps });
    sender.send(RawMessage::Response(resp));
    match receiver.recv() {
        Some(RawMessage::Notification(n)) => {
            n.cast::<Initialized>()
                .map_err(|_| format_err!("expected initialized notification"))?;
        }
        _ => bail!("expected initialized notification"),
    }
    Ok(params)
}
