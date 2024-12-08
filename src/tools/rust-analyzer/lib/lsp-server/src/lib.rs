//! A language server scaffold, exposing a synchronous crossbeam-channel based API.
//! This crate handles protocol handshaking and parsing messages, while you
//! control the message dispatch loop yourself.
//!
//! Run with `RUST_LOG=lsp_server=debug` to see all the messages.

#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(clippy::print_stdout, clippy::disallowed_types)]

mod error;
mod msg;
mod req_queue;
mod socket;
mod stdio;

use std::{
    io,
    net::{TcpListener, TcpStream, ToSocketAddrs},
};

use crossbeam_channel::{Receiver, RecvError, RecvTimeoutError, Sender};

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
        self.initialize_start_while(|| true)
    }

    /// Starts the initialization process by waiting for an initialize as described in
    /// [`Self::initialize_start`] as long as `running` returns
    /// `true` while the return value can be changed through a sig handler such as `CTRL + C`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::atomic::{AtomicBool, Ordering};
    /// use std::sync::Arc;
    /// # use std::error::Error;
    /// # use lsp_types::{ClientCapabilities, InitializeParams, ServerCapabilities};
    /// # use lsp_server::{Connection, Message, Request, RequestId, Response};
    /// # fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    /// let running = Arc::new(AtomicBool::new(true));
    /// # running.store(true, Ordering::SeqCst);
    /// let r = running.clone();
    ///
    /// ctrlc::set_handler(move || {
    ///     r.store(false, Ordering::SeqCst);
    /// }).expect("Error setting Ctrl-C handler");
    ///
    /// let (connection, io_threads) = Connection::stdio();
    ///
    /// let res = connection.initialize_start_while(|| running.load(Ordering::SeqCst));
    /// # assert!(res.is_err());
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn initialize_start_while<C>(
        &self,
        running: C,
    ) -> Result<(RequestId, serde_json::Value), ProtocolError>
    where
        C: Fn() -> bool,
    {
        while running() {
            let msg = match self.receiver.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(msg) => msg,
                Err(RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => return Err(ProtocolError::disconnected()),
            };

            match msg {
                Message::Request(req) if req.is_initialize() => return Ok((req.id, req.params)),
                // Respond to non-initialize requests with ServerNotInitialized
                Message::Request(req) => {
                    let resp = Response::new_err(
                        req.id.clone(),
                        ErrorCode::ServerNotInitialized as i32,
                        format!("expected initialize request, got {req:?}"),
                    );
                    self.sender.send(resp.into()).unwrap();
                    continue;
                }
                Message::Notification(n) if !n.is_exit() => {
                    continue;
                }
                msg => {
                    return Err(ProtocolError::new(format!(
                        "expected initialize request, got {msg:?}"
                    )));
                }
            };
        }

        Err(ProtocolError::new(String::from(
            "Initialization has been aborted during initialization",
        )))
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
            Ok(Message::Notification(n)) if n.is_initialized() => Ok(()),
            Ok(msg) => Err(ProtocolError::new(format!(
                r#"expected initialized notification, got: {msg:?}"#
            ))),
            Err(RecvError) => Err(ProtocolError::disconnected()),
        }
    }

    /// Finishes the initialization process as described in [`Self::initialize_finish`] as
    /// long as `running` returns `true` while the return value can be changed through a sig
    /// handler such as `CTRL + C`.
    pub fn initialize_finish_while<C>(
        &self,
        initialize_id: RequestId,
        initialize_result: serde_json::Value,
        running: C,
    ) -> Result<(), ProtocolError>
    where
        C: Fn() -> bool,
    {
        let resp = Response::new_ok(initialize_id, initialize_result);
        self.sender.send(resp.into()).unwrap();

        while running() {
            let msg = match self.receiver.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(msg) => msg,
                Err(RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ProtocolError::disconnected());
                }
            };

            match msg {
                Message::Notification(n) if n.is_initialized() => {
                    return Ok(());
                }
                msg => {
                    return Err(ProtocolError::new(format!(
                        r#"expected initialized notification, got: {msg:?}"#
                    )));
                }
            }
        }

        Err(ProtocolError::new(String::from(
            "Initialization has been aborted during initialization",
        )))
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

    /// Initialize the connection as described in [`Self::initialize`] as long as `running` returns
    /// `true` while the return value can be changed through a sig handler such as `CTRL + C`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::atomic::{AtomicBool, Ordering};
    /// use std::sync::Arc;
    /// # use std::error::Error;
    /// # use lsp_types::ServerCapabilities;
    /// # use lsp_server::{Connection, Message, Request, RequestId, Response};
    ///
    /// # fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    /// let running = Arc::new(AtomicBool::new(true));
    /// # running.store(true, Ordering::SeqCst);
    /// let r = running.clone();
    ///
    /// ctrlc::set_handler(move || {
    ///     r.store(false, Ordering::SeqCst);
    /// }).expect("Error setting Ctrl-C handler");
    ///
    /// let (connection, io_threads) = Connection::stdio();
    ///
    /// let server_capabilities = serde_json::to_value(&ServerCapabilities::default()).unwrap();
    /// let initialization_params = connection.initialize_while(
    ///     server_capabilities,
    ///     || running.load(Ordering::SeqCst)
    /// );
    ///
    /// # assert!(initialization_params.is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn initialize_while<C>(
        &self,
        server_capabilities: serde_json::Value,
        running: C,
    ) -> Result<serde_json::Value, ProtocolError>
    where
        C: Fn() -> bool,
    {
        let (id, params) = self.initialize_start_while(&running)?;

        let initialize_data = serde_json::json!({
            "capabilities": server_capabilities,
        });

        self.initialize_finish_while(id, initialize_data, running)?;

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
                return Err(ProtocolError::new(format!(
                    "unexpected message during shutdown: {msg:?}"
                )))
            }
            Err(RecvTimeoutError::Timeout) => {
                return Err(ProtocolError::new(
                    "timed out waiting for exit notification".to_owned(),
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(ProtocolError::new(
                    "channel disconnected waiting for exit notification".to_owned(),
                ))
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::unbounded;
    use lsp_types::notification::{Exit, Initialized, Notification};
    use lsp_types::request::{Initialize, Request};
    use lsp_types::{InitializeParams, InitializedParams};
    use serde_json::to_value;

    use crate::{Connection, Message, ProtocolError, RequestId};

    struct TestCase {
        test_messages: Vec<Message>,
        expected_resp: Result<(RequestId, serde_json::Value), ProtocolError>,
    }

    fn initialize_start_test(test_case: TestCase) {
        let (reader_sender, reader_receiver) = unbounded::<Message>();
        let (writer_sender, writer_receiver) = unbounded::<Message>();
        let conn = Connection { sender: writer_sender, receiver: reader_receiver };

        for msg in test_case.test_messages {
            assert!(reader_sender.send(msg).is_ok());
        }

        let resp = conn.initialize_start();
        assert_eq!(test_case.expected_resp, resp);

        assert!(writer_receiver.recv_timeout(std::time::Duration::from_secs(1)).is_err());
    }

    #[test]
    fn not_exit_notification() {
        let notification = crate::Notification {
            method: Initialized::METHOD.to_owned(),
            params: to_value(InitializedParams {}).unwrap(),
        };

        let params_as_value = to_value(InitializeParams::default()).unwrap();
        let req_id = RequestId::from(234);
        let request = crate::Request {
            id: req_id.clone(),
            method: Initialize::METHOD.to_owned(),
            params: params_as_value.clone(),
        };

        initialize_start_test(TestCase {
            test_messages: vec![notification.into(), request.into()],
            expected_resp: Ok((req_id, params_as_value)),
        });
    }

    #[test]
    fn exit_notification() {
        let notification =
            crate::Notification { method: Exit::METHOD.to_owned(), params: to_value(()).unwrap() };
        let notification_msg = Message::from(notification);

        initialize_start_test(TestCase {
            test_messages: vec![notification_msg.clone()],
            expected_resp: Err(ProtocolError::new(format!(
                "expected initialize request, got {notification_msg:?}"
            ))),
        });
    }
}
