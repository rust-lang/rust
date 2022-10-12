//! A language server scaffold, exposing a synchronous crossbeam-channel based API.
//! This crate handles protocol handshaking and parsing messages, while you
//! control the message dispatch loop yourself.
//!
//! Run with `RUST_LOG=lsp_server=debug` to see all the messages.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod msg;
mod stdio;
mod error;
mod socket;
mod req_queue;

use std::{
    io,
    net::{TcpListener, TcpStream, ToSocketAddrs},
};

use crossbeam_channel::{Receiver, Sender};

pub use crate::{
    error::{ExtractError, ProtocolError},
    msg::{ErrorCode, Message, Notification, Request, RequestId, Response, ResponseError},
    req_queue::{Incoming, Outgoing, ReqQueue},
    stdio::IoThreads,
};

/// Connection is just a pair of channels of LSP messages.
pub struct Connection {
    pub sender: Sender<Message>,
    pub receiver: Receiver<Message>,
}

impl Connection {
    /// Create connection over standard in/standard out.
    ///
    /// Use this to create a real language server.
    pub fn stdio() -> (Connection, IoThreads) {
        let (sender, receiver, io_threads) = stdio::stdio_transport();
        (Connection { sender, receiver }, io_threads)
    }

    /// Open a connection over tcp.
    /// This call blocks until a connection is established.
    ///
    /// Use this to create a real language server.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> io::Result<(Connection, IoThreads)> {
        let stream = TcpStream::connect(addr)?;
        let (sender, receiver, io_threads) = socket::socket_transport(stream);
        Ok((Connection { sender, receiver }, io_threads))
    }

    /// Listen for a connection over tcp.
    /// This call blocks until a connection is established.
    ///
    /// Use this to create a real language server.
    pub fn listen<A: ToSocketAddrs>(addr: A) -> io::Result<(Connection, IoThreads)> {
        let listener = TcpListener::bind(addr)?;
        let (stream, _) = listener.accept()?;
        let (sender, receiver, io_threads) = socket::socket_transport(stream);
        Ok((Connection { sender, receiver }, io_threads))
    }

    /// Creates a pair of connected connections.
    ///
    /// Use this for testing.
    pub fn memory() -> (Connection, Connection) {
        let (s1, r1) = crossbeam_channel::unbounded();
        let (s2, r2) = crossbeam_channel::unbounded();
        (Connection { sender: s1, receiver: r2 }, Connection { sender: s2, receiver: r1 })
    }

    /// Starts the initialization process by waiting for an initialize
    /// request from the client. Use this for more advanced customization than
    /// `initialize` can provide.
    ///
    /// Returns the request id and serialized `InitializeParams` from the client.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::error::Error;
    /// use lsp_types::{ClientCapabilities, InitializeParams, ServerCapabilities};
    ///
    /// use lsp_server::{Connection, Message, Request, RequestId, Response};
    ///
    /// fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    ///    // Create the transport. Includes the stdio (stdin and stdout) versions but this could
    ///    // also be implemented to use sockets or HTTP.
    ///    let (connection, io_threads) = Connection::stdio();
    ///
    ///    // Run the server
    ///    let (id, params) = connection.initialize_start()?;
    ///
    ///    let init_params: InitializeParams = serde_json::from_value(params).unwrap();
    ///    let client_capabilities: ClientCapabilities = init_params.capabilities;
    ///    let server_capabilities = ServerCapabilities::default();
    ///
    ///    let initialize_data = serde_json::json!({
    ///        "capabilities": server_capabilities,
    ///        "serverInfo": {
    ///            "name": "lsp-server-test",
    ///            "version": "0.1"
    ///        }
    ///    });
    ///
    ///    connection.initialize_finish(id, initialize_data)?;
    ///
    ///    // ... Run main loop ...
    ///
    ///    Ok(())
    /// }
    /// ```
    pub fn initialize_start(&self) -> Result<(RequestId, serde_json::Value), ProtocolError> {
        loop {
            match self.receiver.recv() {
                Ok(Message::Request(req)) if req.is_initialize() => {
                    return Ok((req.id, req.params))
                }
                // Respond to non-initialize requests with ServerNotInitialized
                Ok(Message::Request(req)) => {
                    let resp = Response::new_err(
                        req.id.clone(),
                        ErrorCode::ServerNotInitialized as i32,
                        format!("expected initialize request, got {:?}", req),
                    );
                    self.sender.send(resp.into()).unwrap();
                }
                Ok(msg) => {
                    return Err(ProtocolError(format!(
                        "expected initialize request, got {:?}",
                        msg
                    )))
                }
                Err(e) => {
                    return Err(ProtocolError(format!(
                        "expected initialize request, got error: {}",
                        e
                    )))
                }
            };
        }
    }

    /// Finishes the initialization process by sending an `InitializeResult` to the client
    pub fn initialize_finish(
        &self,
        initialize_id: RequestId,
        initialize_result: serde_json::Value,
    ) -> Result<(), ProtocolError> {
        let resp = Response::new_ok(initialize_id, initialize_result);
        self.sender.send(resp.into()).unwrap();
        match &self.receiver.recv() {
            Ok(Message::Notification(n)) if n.is_initialized() => (),
            Ok(msg) => {
                return Err(ProtocolError(format!(
                    "expected Message::Notification, got: {:?}",
                    msg,
                )))
            }
            Err(e) => {
                return Err(ProtocolError(format!(
                    "expected initialized notification, got error: {}",
                    e,
                )))
            }
        }
        Ok(())
    }

    /// Initialize the connection. Sends the server capabilities
    /// to the client and returns the serialized client capabilities
    /// on success. If more fine-grained initialization is required use
    /// `initialize_start`/`initialize_finish`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::error::Error;
    /// use lsp_types::ServerCapabilities;
    ///
    /// use lsp_server::{Connection, Message, Request, RequestId, Response};
    ///
    /// fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    ///    // Create the transport. Includes the stdio (stdin and stdout) versions but this could
    ///    // also be implemented to use sockets or HTTP.
    ///    let (connection, io_threads) = Connection::stdio();
    ///
    ///    // Run the server
    ///    let server_capabilities = serde_json::to_value(&ServerCapabilities::default()).unwrap();
    ///    let initialization_params = connection.initialize(server_capabilities)?;
    ///
    ///    // ... Run main loop ...
    ///
    ///    Ok(())
    /// }
    /// ```
    pub fn initialize(
        &self,
        server_capabilities: serde_json::Value,
    ) -> Result<serde_json::Value, ProtocolError> {
        let (id, params) = self.initialize_start()?;

        let initialize_data = serde_json::json!({
            "capabilities": server_capabilities,
        });

        self.initialize_finish(id, initialize_data)?;

        Ok(params)
    }

    /// If `req` is `Shutdown`, respond to it and return `true`, otherwise return `false`
    pub fn handle_shutdown(&self, req: &Request) -> Result<bool, ProtocolError> {
        if !req.is_shutdown() {
            return Ok(false);
        }
        let resp = Response::new_ok(req.id.clone(), ());
        let _ = self.sender.send(resp.into());
        match &self.receiver.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(Message::Notification(n)) if n.is_exit() => (),
            Ok(msg) => {
                return Err(ProtocolError(format!("unexpected message during shutdown: {:?}", msg)))
            }
            Err(e) => {
                return Err(ProtocolError(format!("unexpected error during shutdown: {}", e)))
            }
        }
        Ok(true)
    }
}
