// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

//! A generic client-server implementation based in part on Erlang's
//! [gen_server](http://www.erlang.org/doc/man/gen_server.html) module.

use std::comm::{SharedPort, SharedChan, PortOne, oneshot};

/// Internal message containing a request sent from a client to a
/// `GenericServer`.
enum Request<T, U> {
    /// Sends a request expecting a `Reply` to be sent from the server.
    Call(T),
    /// Sends a request without expecting a reply.
    Cast(U),
}

/// A token for indicating that the server should continue. This is used in
/// `GenericServer::{pre_handle, post_handle}`.
pub struct Continue;

/// A reply message sent from a `GenericServer`.
pub enum Reply<V, StopReason> {
    /// A value to be sent back to the client.
    Reply(V),
    /// This causes the server to close, calling `GenericServer::terminate` with
    /// the supplied reason.
    Stop(StopReason),
}

/// A standardized trait for the definition of servers. These methods
/// are used during the life of the server, when `spawn_server` has been
/// called. The trait implementer usually stores the internal state of the
/// server and can be updated via the methods because they take a `&mut self`.
///
/// # Type parameters
///
/// * `InitArgs` - The type of the argument used during initialization.
/// * `InitError` - An error type that could be returned on the initialization of
/// the server. In order to provide informative error messages, this is required
/// to implement `ToStr`.
/// * `T` - The type of the value sent by `call` to the server.
/// * `U` - The type of the value sent by `cast` to the server.
/// * `V` - The type of the value sent back to the client via a `Client`.
/// * `StopReason` - A reason given for the termination of a server.

// FIXME (#5527): The number of type arguments could be reduced if we had
// associated items.
pub trait GenericServer<InitArgs: Send, InitError: Send + ToStr,
                        T: Send, U: Send, V: Send, StopReason: Send> {
    /// Initialize the server state with the supplied arguments.
    fn init(args: InitArgs) -> Result<Self, InitError>;

    /// Called before the call and cast handlers have been called.
    ///
    /// # Default implementation
    ///
    /// The default implementation always returns `Reply(Continue)`, so
    /// implementing this method is optional.
    ///
    /// # Return value
    ///
    /// - `Reply(Continue)`: to indicate that the server loop should continue.
    /// - `Stop(reason)`: to stop the server loop and call
    ///   `GenericServer::terminate`.
    #[inline]
    fn pre_handle(&mut self) -> Reply<Continue, StopReason> {
        Reply(Continue)
    }

    /// Called after the call and cast handlers have been called.
    ///
    /// # Default implementation
    ///
    /// The default implementation always returns `Reply(Continue)`, so
    /// implementing this method is optional.
    ///
    /// # Return value
    ///
    /// - `Reply(Continue)`: to indicate that the server loop should continue.
    /// - `Stop(reason)`: to stop the server loop before the handlers are called
    ///   and call `GenericServer::terminate`.
    #[inline]
    fn post_handle(&mut self) -> Reply<Continue, StopReason> {
        Reply(Continue)
    }

    /// Called when a a message from `Client::{call, try_call}` was
    /// received. The reply returned instructs the server task to either send
    /// a reply of type `V` back to the client or stop.
    fn handle_call(&mut self, request: T) -> Reply<V, StopReason>;

    /// Called when a a message from `Client::{cast, try_cast}` was
    /// received.
    ///
    /// # Default implementation
    ///
    /// The default implementation always returns `Reply(Continue)`, so
    /// implementing this method is optional.
    ///
    /// # Return value
    ///
    /// - `Reply(Continue)`: to indicate that the server loop should continue.
    /// - `Stop(reason)`: to stop the server loop and call
    ///   `GenericServer::terminate`.
    #[inline] #[allow(unused_variable)]
    fn handle_cast(&mut self, request: U) -> Reply<Continue, StopReason> {
        Reply(Continue)
    }

    /// Called when a `Stop` message was received from
    /// `GenericServer::handle_call`.
    ///
    /// # Default implementation
    ///
    /// Implementing this method is optional - the default implementation
    /// performs no action.
    #[inline] #[allow(unused_variable)]
    fn terminate(self, reason: StopReason) {}
}

/// Initialize a `GenericServer` in a new task.
///
/// # Note
///
/// There are quite a few type parameters required to make this work, so it is
/// probably a good idea to wrap this in a custom constructor method. For
/// example:
///
/// ~~~rust
/// impl MyServer {
///     pub fn spawn() -> Result<Client<(), uint>, ~str>> {
///         spawn::<(), E, X, Y, Z, R, MyServer>(())
///     }
/// }
/// ~~~

// FIXME (#5527): The number of type arguments could be reduced if we had
// associated items.
pub fn spawn_server<InitArgs: Send, InitError: Send + ToStr,
                    T: Send, U: Send, V: Send, StopReason: Send,
                    Server: GenericServer<InitArgs, InitError, T, U, V, StopReason>
                   >(args: InitArgs) -> Result<Client<T, U, V>, InitError> {
    match make_server::<InitArgs, InitError, T, U, V, StopReason, Server>() {
        (spawn_proc, client_port) => {
            spawn(proc() spawn_proc(args)); client_port.recv()
        }
    }
}

/// Returns a `proc` that initializes a server on the current task. This is
/// useful for having more control over what task the server is initialized on.
///
/// # Return value
///
/// For the destructured tuple `(spawn_proc, client_port)`:
///
/// - `spawn_proc`: This `proc` begins the server loop on the current task. The
///   loop continues until `Stop(reason)` has been returned by `pre_handle`,
///   `post_handle`, `handle_call`, or `handle_cast`.
/// - `client_port`: This oneshot port returns a `Client` that can communicate
///   with the `GenericServer`.
///
/// # Examples
///
/// This shows how a `make_server` can be used to duplicate the functionality
/// of the `spawn_server` function:
///
/// ~~~rust
/// let client = match make_server::<(), E, X, Y, Z, R, MyServer>() {
///     (spawn_proc, client_port) => {
///         spawn(proc() spawn_proc(())); client_port.recv()
///     }
/// };
/// ~~~
///
/// This shows how `make_server` can be used to spawn a server directly on the
/// main thread. This is useful for wrapping C libraries that need direct access
/// to the main event loop.
///
/// ~~~rust
/// #[start]
/// fn start(argc: int, argv: **u8) -> int {
///     std::rt::start_on_main_thread(argc, argv, main)
/// }
///
/// fn main() {
///     let (spawn_proc, client_port) =
///         make_server::<(), E, X, Y, Z, R, MyServer>(());
///
///     spawn(proc() {
///         let client = client_port.recv();
///         // do client things ...
///     });
///
///     spawn_proc(());
/// }
/// ~~~

// FIXME (#5527): The number of type arguments could be reduced if we had
// associated items.
pub fn make_server<InitArgs: Send, InitError: Send + ToStr,
                   T: Send, U: Send, V: Send, StopReason: Send,
                   Server: GenericServer<InitArgs, InitError, T, U, V, StopReason>
                  >() -> (proc(InitArgs), PortOne<Result<Client<T, U, V>, InitError>>) {
    // A oneshot stream for returning the server handle from the server task
    let (return_port, return_chan) = oneshot();
    (proc(args: InitArgs) {
        let res: Result<Server, InitError> = GenericServer::init(args);
        match res {
            Ok(mut server) => {
                // Initialize some streams for communicating with the clients.
                let (request_port, request_chan) = stream::<Request<T, U>>();
                let (reply_port, reply_chan) = stream::<V>();

                // Returns a server handle via the port
                return_chan.send(Ok(Client {
                    request_chan: request_chan,
                    reply_port: reply_port,
                }));

                // Begin recieving messages from the server handles.
                for request in request_port.recv_iter() {
                    macro_rules! handle_stop(
                        ($s:expr, $e:expr) => (match $e {
                            Stop(reason) => { $s.terminate(reason); break; }
                            Reply(x) => x,
                        })
                    )

                    // Do pre-handling operations
                    handle_stop!(server, server.pre_handle());
                    // Handle `Call` and `Cast` requests
                    match request {
                        Call(x) => {
                            let reply = handle_stop!(server, server.handle_call(x));
                            reply_chan.send(reply);
                        }
                        Cast(x) => {
                            handle_stop!(server, server.handle_cast(x));
                        }
                    }
                    // Do post-handling operations
                    handle_stop!(server, server.post_handle());
                }
            }
            Err(e) => {
                return_chan.send(Err(e));
            }
        }
    }, return_port)
}

pub trait GenericClient<T: Send, U: Send, V: Send> {
    /// Makes a synchronous call to the `GenericServer` linked to the client,
    /// blocking until a reply has been received.
    fn call(&self, request: T) -> V;

    /// Makes a synchronous call the `GenericServer`, returning `None` if the
    /// server was closed before the reply could be received.
    fn try_call(&self, request: T) -> Option<V>;

    // FIXME (#000): add call_with_timeout.

    /// Makes a synchronous request to the `GenericServer` linked to the handle
    /// without expecting a reply.
    fn cast(&self, request: U);

    /// Makes a synchronous request to the `GenericServer`linked to the handle
    /// without expecting a reply, returning `true` if the request was received
    /// successfully, or `false` if the server was closed before the message was
    /// handled.
    fn try_cast(&self, request: U) -> bool;

    // FIXME (#000): add cast_with_timeout.
}

macro_rules! impl_client(
    ($Self:ident) => (impl<T: Send, U: Send, V: Send> GenericClient<T, U, V> for $Self<T, U, V> {
        fn call(&self, request: T) -> V {
            self.try_call(request).expect("GenericClient::call: server closed.")
        }

        fn try_call(&self, request: T) -> Option<V> {
            if !self.request_chan.try_send(Call(request)) { None }
            else { self.reply_port.try_recv() }
        }

        fn cast(&self, request: U) {
            self.request_chan.send(Cast(request));
        }

        fn try_cast(&self, request: U) -> bool {
            self.request_chan.try_send(Cast(request))
        }
    })
)

/// A client for communicating with a server
pub struct Client<T, U, V> {
    priv request_chan: Chan<Request<T, U>>,
    priv reply_port: Port<V>,
}

impl<T: Send, U: Send, V: Send> Client<T, U, V> {
    pub fn into_shared(self) -> SharedClient<T, U, V> {
        SharedClient::new(self)
    }
}

impl_client!(Client)

/// A client that can be shared between tasks
pub struct SharedClient<T, U, V> {
    priv request_chan: SharedChan<Request<T, U>>,
    priv reply_port: SharedPort<V>,
}

impl<T: Send, U: Send, V: Send> SharedClient<T, U, V> {
    pub fn new(Client { request_chan, reply_port }: Client<T, U, V>)
               -> SharedClient<T, U, V> {
        SharedClient {
            request_chan: SharedChan::new(request_chan),
            reply_port: SharedPort::new(reply_port),
        }
    }
}

impl_client!(SharedClient)

impl<T: Send, U: Send, V: Send> Clone for SharedClient<T, U, V> {
    fn clone(&self) -> SharedClient<T, U, V> {
        SharedClient {
            reply_port: self.reply_port.clone(),
            request_chan: self.request_chan.clone(),
        }
    }
}

#[cfg(test)]
mod test_cell {
    //! This test demonstrates the implementation of an updatable storage cell
    //! using the `GenericServer` trait. This is based off the cell example
    //! described in _Concurrent Programming in ML_ (Reppy, 1999, pp. 42-45).

    use super::*;

    /// The storage server for the cell.
    struct Server { value: int }

    struct Get;
    struct Put(int);

    impl GenericServer<int, (), Get, Put, int, ()> for Server {
        fn init(x: int) -> Result<Server, ()> {
            Ok(Server { value: x })
        }

        fn handle_call(&mut self, _: Get) -> Reply<int, ()> {
            Reply(self.value)
        }

        fn handle_cast(&mut self, Put(x): Put) -> Reply<Continue, ()> {
            self.value = x; Reply(Continue)
        }
    }

    struct Cell {
        priv client: SharedClient<Get, Put, int>,
    }

    impl Cell {
        /// Create a new storage cell from the initial value `x`.
        fn new(x: int) -> Cell {
            Cell {
                client: spawn_server::<int, (), Get, Put, int, (), Server>(x)
                    .unwrap().into_shared()
            }
        }

        /// Read the contents of the cell.
        fn get(&self) -> int { self.client.call(Get) }

        /// Update the contents of the cell.
        fn put(&self, x: int) { self.client.cast(Put(x)) }
    }

    impl Clone for Cell {
        fn clone(&self) -> Cell {
            Cell { client: self.client.clone() }
        }
    }

    #[test]
    fn test_new_get_put() {
        let cell = Cell::new(1);
        assert_eq!(cell.get(), 1);
        cell.put(2);
        assert_eq!(cell.get(), 2);
        assert_eq!(cell.get(), 2);

        let cell2 = cell.clone();
        assert_eq!(cell.get(), 2);
        cell2.put(3);
        assert_eq!(cell.get(), 3);
        assert_eq!(cell2.get(), 3);
    }
}
