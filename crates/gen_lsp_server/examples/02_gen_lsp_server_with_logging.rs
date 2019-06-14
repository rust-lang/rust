//! A minimal example LSP server that can only respond to the `gotoDefinition` request. To use
//! this example, execute it and then send an `initialize` request.
//!
//! ```no_run
//! Content-Length: 85
//!
//! {"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {"capabilities": {}}}
//! ```
//!
//! This will respond with a server respose. Then send it a `initialized` notification which will
//! have no response.
//!
//! ```no_run
//! Content-Length: 59
//!
//! {"jsonrpc": "2.0", "method": "initialized", "params": {}}
//! ```
//!
//! Once these two are sent, then we enter the main loop of the server. The only request this
//! example can handle is `gotoDefinition`:
//!
//! ```no_run
//! Content-Length: 159
//!
//! {"jsonrpc": "2.0", "method": "textDocument/definition", "id": 2, "params": {"textDocument": {"uri": "file://temp"}, "position": {"line": 1, "character": 1}}}
//! ```
//!
//! To finish up without errors, send a shutdown request:
//!
//! ```no_run
//! Content-Length: 67
//!
//! {"jsonrpc": "2.0", "method": "shutdown", "id": 3, "params": null}
//! ```
//!
//! The server will exit the main loop and finally we send a `shutdown` notification to stop
//! the server.
//!
//! ```
//! Content-Length: 54
//!
//! {"jsonrpc": "2.0", "method": "exit", "params": null}
//! ```

use std::error::Error;

use crossbeam_channel::{Sender, Receiver};
use lsp_types::{
    ServerCapabilities, InitializeParams,
    request::{GotoDefinition, GotoDefinitionResponse},
};
use log::info;
use gen_lsp_server::{
    run_server, stdio_transport, handle_shutdown, RawMessage, RawResponse, RawRequest,
};

fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    // Set up logging. Because `stdio_transport` gets a lock on stdout and stdin, we must have
    // our logging only write out to stderr.
    flexi_logger::Logger::with_str("info").start().unwrap();
    info!("starting generic LSP server");

    // Create the transport. Includes the stdio (stdin and stdout) versions but this could
    // also be implemented to use sockets or HTTP.
    let (receiver, sender, io_threads) = stdio_transport();

    // Run the server and wait for the two threads to end (typically by trigger LSP Exit event).
    run_server(ServerCapabilities::default(), receiver, sender, main_loop)?;
    io_threads.join()?;

    // Shut down gracefully.
    info!("shutting down server");
    Ok(())
}

fn main_loop(
    _params: InitializeParams,
    receiver: &Receiver<RawMessage>,
    sender: &Sender<RawMessage>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    info!("starting example main loop");
    for msg in receiver {
        info!("got msg: {:?}", msg);
        match msg {
            RawMessage::Request(req) => {
                let req = match log_handle_shutdown(req, sender) {
                    None => return Ok(()),
                    Some(req) => req,
                };
                info!("got request: {:?}", req);
                match req.cast::<GotoDefinition>() {
                    Ok((id, params)) => {
                        info!("got gotoDefinition request #{}: {:?}", id, params);
                        let resp = RawResponse::ok::<GotoDefinition>(
                            id,
                            &Some(GotoDefinitionResponse::Array(Vec::new())),
                        );
                        info!("sending gotoDefinition response: {:?}", resp);
                        sender.send(RawMessage::Response(resp))?;
                        continue;
                    }
                    Err(req) => req,
                };
                // ...
            }
            RawMessage::Response(resp) => {
                info!("got response: {:?}", resp);
            }
            RawMessage::Notification(not) => {
                info!("got notification: {:?}", not);
            }
        }
    }
    Ok(())
}

pub fn log_handle_shutdown(req: RawRequest, sender: &Sender<RawMessage>) -> Option<RawRequest> {
    info!("handle_shutdown: {:?}", req);
    handle_shutdown(req, sender)
}
