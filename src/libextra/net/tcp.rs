// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! High-level interface to libuv's TCP functionality
// FIXME #4425: Need FFI fixes

#[allow(missing_doc)];


use future;
use future_spawn = future::spawn;
use ip = net::ip;
use uv;
use uv::iotask;
use uv::iotask::IoTask;

use std::io;
use std::libc::size_t;
use std::libc;
use std::comm::{stream, Port, SharedChan};
use std::ptr;
use std::result::{Result};
use std::result;
use std::num;
use std::vec;

pub mod rustrt {
    use std::libc;

    #[nolink]
    pub extern {
        unsafe fn rust_uv_current_kernel_malloc(size: libc::c_uint)
                                             -> *libc::c_void;
        unsafe fn rust_uv_current_kernel_free(mem: *libc::c_void);
        unsafe fn rust_uv_helper_uv_tcp_t_size() -> libc::c_uint;
    }
}

/**
 * Encapsulates an open TCP/IP connection through libuv
 *
 * `TcpSocket` is non-copyable/sendable and automagically handles closing the
 * underlying libuv data structures when it goes out of scope. This is the
 * data structure that is used for read/write operations over a TCP stream.
 */
pub struct TcpSocket {
  socket_data: @TcpSocketData,
}

#[unsafe_destructor]
impl Drop for TcpSocket {
    fn drop(&self) {
        tear_down_socket_data(self.socket_data)
    }
}

pub fn TcpSocket(socket_data: @TcpSocketData) -> TcpSocket {
    TcpSocket {
        socket_data: socket_data
    }
}

/**
 * A buffered wrapper for `net::tcp::TcpSocket`
 *
 * It is created with a call to `net::tcp::socket_buf()` and has impls that
 * satisfy both the `io::Reader` and `io::Writer` traits.
 */
pub struct TcpSocketBuf {
    data: @mut TcpBufferedSocketData,
    end_of_stream: @mut bool
}

pub fn TcpSocketBuf(data: @mut TcpBufferedSocketData) -> TcpSocketBuf {
    TcpSocketBuf {
        data: data,
        end_of_stream: @mut false
    }
}

/// Contains raw, string-based, error information returned from libuv
#[deriving(Clone)]
pub struct TcpErrData {
    err_name: ~str,
    err_msg: ~str,
}

/// Details returned as part of a `Result::Err` result from `tcp::listen`
pub enum TcpListenErrData {
    /**
     * Some unplanned-for error. The first and second fields correspond
     * to libuv's `err_name` and `err_msg` fields, respectively.
     */
    GenericListenErr(~str, ~str),
    /**
     * Failed to bind to the requested IP/Port, because it is already in use.
     *
     * # Possible Causes
     *
     * * Attempting to bind to a port already bound to another listener
     */
    AddressInUse,
    /**
     * Request to bind to an IP/Port was denied by the system.
     *
     * # Possible Causes
     *
     * * Attemping to binding to an IP/Port as a non-Administrator
     *   on Windows Vista+
     * * Attempting to bind, as a non-priv'd
     *   user, to 'privileged' ports (< 1024) on *nix
     */
    AccessDenied
}
/// Details returned as part of a `Result::Err` result from `tcp::connect`
pub enum TcpConnectErrData {
    /**
     * Some unplanned-for error. The first and second fields correspond
     * to libuv's `err_name` and `err_msg` fields, respectively.
     */
    GenericConnectErr(~str, ~str),
    /// Invalid IP or invalid port
    ConnectionRefused
}

/**
 * Initiate a client connection over TCP/IP
 *
 * # Arguments
 *
 * * `input_ip` - The IP address (versions 4 or 6) of the remote host
 * * `port` - the unsigned integer of the desired remote host port
 * * `iotask` - a `uv::iotask` that the tcp request will run on
 *
 * # Returns
 *
 * A `result` that, if the operation succeeds, contains a
 * `net::net::TcpSocket` that can be used to send and receive data to/from
 * the remote host. In the event of failure, a
 * `net::tcp::TcpConnectErrData` instance will be returned
 */
pub fn connect(input_ip: ip::IpAddr, port: uint,
               iotask: &IoTask)
    -> result::Result<TcpSocket, TcpConnectErrData> {
    unsafe {
        let (result_po, result_ch) = stream::<ConnAttempt>();
        let result_ch = SharedChan::new(result_ch);
        let (closed_signal_po, closed_signal_ch) = stream::<()>();
        let closed_signal_ch = SharedChan::new(closed_signal_ch);
        let conn_data = ConnectReqData {
            result_ch: result_ch,
            closed_signal_ch: closed_signal_ch
        };
        let conn_data_ptr: *ConnectReqData = &conn_data;
        let (reader_po, reader_ch) = stream::<Result<~[u8], TcpErrData>>();
        let reader_ch = SharedChan::new(reader_ch);
        let stream_handle_ptr = malloc_uv_tcp_t();
        *(stream_handle_ptr as *mut uv::ll::uv_tcp_t) = uv::ll::tcp_t();
        let socket_data = @TcpSocketData {
            reader_po: @reader_po,
            reader_ch: reader_ch,
            stream_handle_ptr: stream_handle_ptr,
            connect_req: uv::ll::connect_t(),
            write_req: uv::ll::write_t(),
            ipv6: match input_ip {
                ip::Ipv4(_) => { false }
                ip::Ipv6(_) => { true }
            },
            iotask: iotask.clone()
        };
        let socket_data_ptr: *TcpSocketData = &*socket_data;
        // get an unsafe representation of our stream_handle_ptr that
        // we can send into the interact cb to be handled in libuv..
        debug!("stream_handle_ptr outside interact %?",
                        stream_handle_ptr);
        do iotask::interact(iotask) |loop_ptr| {
            debug!("in interact cb for tcp client connect..");
            debug!("stream_handle_ptr in interact %?",
                            stream_handle_ptr);
            match uv::ll::tcp_init( loop_ptr, stream_handle_ptr) {
                0i32 => {
                    debug!("tcp_init successful");
                    debug!("dealing w/ ipv4 connection..");
                    let connect_req_ptr: *uv::ll::uv_connect_t =
                        &(*socket_data_ptr).connect_req;
                    let addr_str = ip::format_addr(&input_ip);
                    let connect_result = match input_ip {
                        ip::Ipv4(ref addr) => {
                            // have to "recreate" the
                            // sockaddr_in/6 since the ip_addr
                            // discards the port info.. should
                            // probably add an additional rust
                            // type that actually is closer to
                            // what the libuv API expects (ip str
                            // + port num)
                            debug!("addr: %?", addr);
                            let in_addr = uv::ll::ip4_addr(addr_str,
                                                           port as int);
                            uv::ll::tcp_connect(
                                connect_req_ptr,
                                stream_handle_ptr,
                                &in_addr,
                                tcp_connect_on_connect_cb)
                        }
                        ip::Ipv6(ref addr) => {
                            debug!("addr: %?", addr);
                            let in_addr = uv::ll::ip6_addr(addr_str,
                                                           port as int);
                            uv::ll::tcp_connect6(
                                connect_req_ptr,
                                stream_handle_ptr,
                                &in_addr,
                                tcp_connect_on_connect_cb)
                        }
                    };
                    match connect_result {
                        0i32 => {
                            debug!("tcp_connect successful: \
                                    stream %x,
                                    socket data %x",
                                   stream_handle_ptr as uint,
                                   socket_data_ptr as uint);
                            // reusable data that we'll have for the
                            // duration..
                            uv::ll::set_data_for_uv_handle(
                                stream_handle_ptr,
                                socket_data_ptr as
                                *libc::c_void);
                            // just so the connect_cb can send the
                            // outcome..
                            uv::ll::set_data_for_req(connect_req_ptr,
                                                     conn_data_ptr);
                            debug!("leaving tcp_connect interact cb...");
                            // let tcp_connect_on_connect_cb send on
                            // the result_ch, now..
                        }
                        _ => {
                            // immediate connect
                            // failure.. probably a garbage ip or
                            // somesuch
                            let err_data =
                                uv::ll::get_last_err_data(loop_ptr);
                            let result_ch = (*conn_data_ptr)
                                .result_ch.clone();
                            result_ch.send(ConnFailure(err_data));
                            uv::ll::set_data_for_uv_handle(
                                stream_handle_ptr,
                                conn_data_ptr);
                            uv::ll::close(stream_handle_ptr,
                                          stream_error_close_cb);
                        }
                    }
                }
                _ => {
                    // failure to create a tcp handle
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    let result_ch = (*conn_data_ptr).result_ch.clone();
                    result_ch.send(ConnFailure(err_data));
                }
            }
        }
        match result_po.recv() {
            ConnSuccess => {
                debug!("tcp::connect - received success on result_po");
                result::Ok(TcpSocket(socket_data))
            }
            ConnFailure(ref err_data) => {
                closed_signal_po.recv();
                debug!("tcp::connect - received failure on result_po");
                // still have to free the malloc'd stream handle..
                rustrt::rust_uv_current_kernel_free(stream_handle_ptr
                                                    as *libc::c_void);
                let tcp_conn_err = match err_data.err_name {
                    ~"ECONNREFUSED" => ConnectionRefused,
                    _ => GenericConnectErr(err_data.err_name.clone(),
                                           err_data.err_msg.clone())
                };
                result::Err(tcp_conn_err)
            }
        }
    }
}

/**
 * Write binary data to a tcp stream; Blocks until operation completes
 *
 * # Arguments
 *
 * * sock - a `TcpSocket` to write to
 * * raw_write_data - a vector of `~[u8]` that will be written to the stream.
 * This value must remain valid for the duration of the `write` call
 *
 * # Returns
 *
 * A `Result` object with a `()` value as the `Ok` variant, or a
 * `TcpErrData` value as the `Err` variant
 */
pub fn write(sock: &TcpSocket, raw_write_data: ~[u8])
             -> result::Result<(), TcpErrData> {
    let socket_data_ptr: *TcpSocketData = &*sock.socket_data;
    write_common_impl(socket_data_ptr, raw_write_data)
}

/**
 * Write binary data to tcp stream; Returns a `future::Future` value
 * immediately
 *
 * # Safety
 *
 * This function can produce unsafe results if:
 *
 * 1. the call to `write_future` is made
 * 2. the `future::Future` value returned is never resolved via
 * `Future::get`
 * 3. and then the `TcpSocket` passed in to `write_future` leaves
 * scope and is destructed before the task that runs the libuv write
 * operation completes.
 *
 * As such: If using `write_future`, always be sure to resolve the returned
 * `Future` so as to ensure libuv doesn't try to access a released write
 * handle. Otherwise, use the blocking `tcp::write` function instead.
 *
 * # Arguments
 *
 * * sock - a `TcpSocket` to write to
 * * raw_write_data - a vector of `~[u8]` that will be written to the stream.
 * This value must remain valid for the duration of the `write` call
 *
 * # Returns
 *
 * A `Future` value that, once the `write` operation completes, resolves to a
 * `Result` object with a `nil` value as the `Ok` variant, or a `TcpErrData`
 * value as the `Err` variant
 */
pub fn write_future(sock: &TcpSocket, raw_write_data: ~[u8])
    -> future::Future<result::Result<(), TcpErrData>>
{
    let socket_data_ptr: *TcpSocketData = &*sock.socket_data;
    do future_spawn {
        let data_copy = raw_write_data.clone();
        write_common_impl(socket_data_ptr, data_copy)
    }
}

/**
 * Begin reading binary data from an open TCP connection; used with
 * `read_stop`
 *
 * # Arguments
 *
 * * sock -- a `net::tcp::TcpSocket` for the connection to read from
 *
 * # Returns
 *
 * * A `Result` instance that will either contain a
 * `std::comm::Port<Result<~[u8], TcpErrData>>` that the user can read
 * (and * optionally, loop on) from until `read_stop` is called, or a
 * `TcpErrData` record
 */
pub fn read_start(sock: &TcpSocket)
                  -> result::Result<@Port<result::Result<~[u8],
                                                         TcpErrData>>,
                                    TcpErrData> {
    let socket_data: *TcpSocketData = &*sock.socket_data;
    read_start_common_impl(socket_data)
}

/**
 * Stop reading from an open TCP connection; used with `read_start`
 *
 * # Arguments
 *
 * * `sock` - a `net::tcp::TcpSocket` that you wish to stop reading on
 */
pub fn read_stop(sock: &TcpSocket) -> result::Result<(), TcpErrData> {
    let socket_data: *TcpSocketData = &*sock.socket_data;
    read_stop_common_impl(socket_data)
}

/**
 * Reads a single chunk of data from `TcpSocket`; block until data/error
 * recv'd
 *
 * Does a blocking read operation for a single chunk of data from a
 * `TcpSocket` until a data arrives or an error is received. The provided
 * `timeout_msecs` value is used to raise an error if the timeout period
 * passes without any data received.
 *
 * # Arguments
 *
 * * `sock` - a `net::tcp::TcpSocket` that you wish to read from
 * * `timeout_msecs` - a `uint` value, in msecs, to wait before dropping the
 * read attempt. Pass `0u` to wait indefinitely
 */
pub fn read(sock: &TcpSocket, timeout_msecs: uint)
            -> result::Result<~[u8],TcpErrData> {
    let socket_data: *TcpSocketData = &*sock.socket_data;
    read_common_impl(socket_data, timeout_msecs)
}

/**
 * Reads a single chunk of data; returns a `future::Future<~[u8]>`
 * immediately
 *
 * Does a non-blocking read operation for a single chunk of data from a
 * `TcpSocket` and immediately returns a `Future` value representing the
 * result. When resolving the returned `Future`, it will block until data
 * arrives or an error is received. The provided `timeout_msecs`
 * value is used to raise an error if the timeout period passes without any
 * data received.
 *
 * # Safety
 *
 * This function can produce unsafe results if the call to `read_future` is
 * made, the `future::Future` value returned is never resolved via
 * `Future::get`, and then the `TcpSocket` passed in to `read_future` leaves
 * scope and is destructed before the task that runs the libuv read
 * operation completes.
 *
 * As such: If using `read_future`, always be sure to resolve the returned
 * `Future` so as to ensure libuv doesn't try to access a released read
 * handle. Otherwise, use the blocking `tcp::read` function instead.
 *
 * # Arguments
 *
 * * `sock` - a `net::tcp::TcpSocket` that you wish to read from
 * * `timeout_msecs` - a `uint` value, in msecs, to wait before dropping the
 * read attempt. Pass `0u` to wait indefinitely
 */
fn read_future(sock: &TcpSocket, timeout_msecs: uint)
               -> future::Future<result::Result<~[u8],TcpErrData>> {
    let socket_data: *TcpSocketData = &*sock.socket_data;
    do future_spawn {
        read_common_impl(socket_data, timeout_msecs)
    }
}

/**
 * Bind an incoming client connection to a `net::tcp::TcpSocket`
 *
 * # Notes
 *
 * It is safe to call `net::tcp::accept` _only_ within the context of the
 * `new_connect_cb` callback provided as the final argument to the
 * `net::tcp::listen` function.
 *
 * The `new_conn` opaque value is provided _only_ as the first argument to the
 * `new_connect_cb` provided as a part of `net::tcp::listen`.
 * It can be safely sent to another task but it _must_ be
 * used (via `net::tcp::accept`) before the `new_connect_cb` call it was
 * provided to returns.
 *
 * This implies that a port/chan pair must be used to make sure that the
 * `new_connect_cb` call blocks until an attempt to create a
 * `net::tcp::TcpSocket` is completed.
 *
 * # Example
 *
 * Here, the `new_conn` is used in conjunction with `accept` from within
 * a task spawned by the `new_connect_cb` passed into `listen`
 *
 * ~~~ {.rust}
 * do net::tcp::listen(remote_ip, remote_port, backlog, iotask,
 *     // this callback is ran once after the connection is successfully
 *     // set up
 *     |kill_ch| {
 *       // pass the kill_ch to your main loop or wherever you want
 *       // to be able to externally kill the server from
 *     })
 *     // this callback is ran when a new connection arrives
 *     |new_conn, kill_ch| {
 *     let (cont_po, cont_ch) = comm::stream::<option::Option<TcpErrData>>();
 *     do task::spawn {
 *         let accept_result = net::tcp::accept(new_conn);
 *         match accept_result {
 *             Err(accept_error) => {
 *                 cont_ch.send(Some(accept_error));
 *                 // fail?
 *             },
 *             Ok(sock) => {
 *                 cont_ch.send(None);
 *                 // do work here
 *             }
 *         }
 *     };
 *     match cont_po.recv() {
 *       // shut down listen()
 *       Some(err_data) => kill_ch.send(Some(err_data)),
 *       // wait for next connection
 *       None => ()
 *     }
 * };
 * ~~~
 *
 * # Arguments
 *
 * * `new_conn` - an opaque value used to create a new `TcpSocket`
 *
 * # Returns
 *
 * On success, this function will return a `net::tcp::TcpSocket` as the
 * `Ok` variant of a `Result`. The `net::tcp::TcpSocket` is anchored within
 * the task that `accept` was called within for its lifetime. On failure,
 * this function will return a `net::tcp::TcpErrData` record
 * as the `Err` variant of a `Result`.
 */
pub fn accept(new_conn: TcpNewConnection)
    -> result::Result<TcpSocket, TcpErrData> {
    unsafe {
        match new_conn{
            NewTcpConn(server_handle_ptr) => {
                let server_data_ptr = uv::ll::get_data_for_uv_handle(
                    server_handle_ptr) as *TcpListenFcData;
                let (reader_po, reader_ch) = stream::<
                    Result<~[u8], TcpErrData>>();
                let reader_ch = SharedChan::new(reader_ch);
                let iotask = &(*server_data_ptr).iotask;
                let stream_handle_ptr = malloc_uv_tcp_t();
                *(stream_handle_ptr as *mut uv::ll::uv_tcp_t) =
                    uv::ll::tcp_t();
                let client_socket_data: @TcpSocketData = @TcpSocketData {
                    reader_po: @reader_po,
                    reader_ch: reader_ch,
                    stream_handle_ptr : stream_handle_ptr,
                    connect_req : uv::ll::connect_t(),
                    write_req : uv::ll::write_t(),
                    ipv6: (*server_data_ptr).ipv6,
                    iotask : iotask.clone()
                };
                let client_socket_data_ptr: *TcpSocketData =
                    &*client_socket_data;
                let client_stream_handle_ptr =
                    (*client_socket_data_ptr).stream_handle_ptr;

                let (result_po, result_ch) = stream::<Option<TcpErrData>>();
                let result_ch = SharedChan::new(result_ch);

                // UNSAFE LIBUV INTERACTION BEGIN
                // .. normally this happens within the context of
                // a call to uv::hl::interact.. but we're breaking
                // the rules here because this always has to be
                // called within the context of a listen() new_connect_cb
                // callback (or it will likely fail and drown your cat)
                debug!("in interact cb for tcp::accept");
                let loop_ptr = uv::ll::get_loop_for_uv_handle(
                    server_handle_ptr);
                match uv::ll::tcp_init(loop_ptr, client_stream_handle_ptr) {
                    0i32 => {
                        debug!("uv_tcp_init successful for \
                                     client stream");
                        match uv::ll::accept(
                            server_handle_ptr as *libc::c_void,
                            client_stream_handle_ptr as *libc::c_void) {
                            0i32 => {
                                debug!("successfully accepted client \
                                        connection: \
                                        stream %x, \
                                        socket data %x",
                                       client_stream_handle_ptr as uint,
                                       client_socket_data_ptr as uint);
                                uv::ll::set_data_for_uv_handle(
                                    client_stream_handle_ptr,
                                    client_socket_data_ptr
                                    as *libc::c_void);
                                let ptr = uv::ll::get_data_for_uv_handle(
                                    client_stream_handle_ptr);
                                debug!("ptrs: %x %x",
                                       client_socket_data_ptr as uint,
                                       ptr as uint);
                                result_ch.send(None);
                            }
                            _ => {
                                debug!("failed to accept client conn");
                                result_ch.send(Some(
                                    uv::ll::get_last_err_data(
                                        loop_ptr).to_tcp_err()));
                            }
                        }
                    }
                    _ => {
                        debug!("failed to accept client stream");
                        result_ch.send(Some(
                            uv::ll::get_last_err_data(
                                loop_ptr).to_tcp_err()));
                    }
                }
                // UNSAFE LIBUV INTERACTION END
                match result_po.recv() {
                    Some(err_data) => result::Err(err_data),
                    None => result::Ok(TcpSocket(client_socket_data))
                }
            }
        }
    }
}

/**
 * Bind to a given IP/port and listen for new connections
 *
 * # Arguments
 *
 * * `host_ip` - a `net::ip::IpAddr` representing a unique IP
 * (versions 4 or 6)
 * * `port` - a uint representing the port to listen on
 * * `backlog` - a uint representing the number of incoming connections
 * to cache in memory
 * * `hl_loop` - a `uv_iotask::IoTask` that the tcp request will run on
 * * `on_establish_cb` - a callback that is evaluated if/when the listener
 * is successfully established. it takes no parameters
 * * `new_connect_cb` - a callback to be evaluated, on the libuv thread,
 * whenever a client attempts to conect on the provided ip/port. the
 * callback's arguments are:
 *     * `new_conn` - an opaque type that can be passed to
 *     `net::tcp::accept` in order to be converted to a `TcpSocket`.
 *     * `kill_ch` - channel of type `std::comm::Chan<Option<tcp_err_data>>`.
 *     this channel can be used to send a message to cause `listen` to begin
 *     closing the underlying libuv data structures.
 *
 * # returns
 *
 * a `Result` instance containing empty data of type `()` on a
 * successful/normal shutdown, and a `TcpListenErrData` enum in the event
 * of listen exiting because of an error
 */
pub fn listen(host_ip: ip::IpAddr, port: uint, backlog: uint,
              iotask: &IoTask,
              on_establish_cb: ~fn(SharedChan<Option<TcpErrData>>),
              new_connect_cb: ~fn(TcpNewConnection,
                                  SharedChan<Option<TcpErrData>>))
    -> result::Result<(), TcpListenErrData> {
    do listen_common(host_ip, port, backlog, iotask,
                     on_establish_cb)
        // on_connect_cb
        |handle| {
        unsafe {
            let server_data_ptr = uv::ll::get_data_for_uv_handle(handle)
                as *TcpListenFcData;
            let new_conn = NewTcpConn(handle);
            let kill_ch = (*server_data_ptr).kill_ch.clone();
            new_connect_cb(new_conn, kill_ch);
        }
    }
}

fn listen_common(host_ip: ip::IpAddr,
                 port: uint,
                 backlog: uint,
                 iotask: &IoTask,
                 on_establish_cb: ~fn(SharedChan<Option<TcpErrData>>),
                 on_connect_cb: ~fn(*uv::ll::uv_tcp_t))
              -> result::Result<(), TcpListenErrData> {
    let (stream_closed_po, stream_closed_ch) = stream::<()>();
    let stream_closed_ch = SharedChan::new(stream_closed_ch);
    let (kill_po, kill_ch) = stream::<Option<TcpErrData>>();
    let kill_ch = SharedChan::new(kill_ch);
    let server_stream = uv::ll::tcp_t();
    let server_stream_ptr: *uv::ll::uv_tcp_t = &server_stream;
    let server_data: TcpListenFcData = TcpListenFcData {
        server_stream_ptr: server_stream_ptr,
        stream_closed_ch: stream_closed_ch,
        kill_ch: kill_ch.clone(),
        on_connect_cb: on_connect_cb,
        iotask: iotask.clone(),
        ipv6: match &host_ip {
            &ip::Ipv4(_) => { false }
            &ip::Ipv6(_) => { true }
        },
        active: @mut true
    };
    let server_data_ptr: *TcpListenFcData = &server_data;

    let (setup_po, setup_ch) = stream();

    // this is to address a compiler warning about
    // an implicit copy.. it seems that double nested
    // will defeat a move sigil, as is done to the host_ip
    // arg above.. this same pattern works w/o complaint in
    // tcp::connect (because the iotask::interact cb isn't
    // nested within a core::comm::listen block)
    let loc_ip = host_ip;
    do iotask::interact(iotask) |loop_ptr| {
        unsafe {
            match uv::ll::tcp_init(loop_ptr, server_stream_ptr) {
                0i32 => {
                    uv::ll::set_data_for_uv_handle(
                        server_stream_ptr,
                        server_data_ptr);
                    let addr_str = ip::format_addr(&loc_ip);
                    let bind_result = match loc_ip {
                        ip::Ipv4(ref addr) => {
                            debug!("addr: %?", addr);
                            let in_addr = uv::ll::ip4_addr(
                                addr_str,
                                port as int);
                            uv::ll::tcp_bind(server_stream_ptr, &in_addr)
                        }
                        ip::Ipv6(ref addr) => {
                            debug!("addr: %?", addr);
                            let in_addr = uv::ll::ip6_addr(
                                addr_str,
                                port as int);
                            uv::ll::tcp_bind6(server_stream_ptr, &in_addr)
                        }
                    };
                    match bind_result {
                        0i32 => {
                            match uv::ll::listen(
                                server_stream_ptr,
                                backlog as libc::c_int,
                                tcp_lfc_on_connection_cb) {
                                0i32 => setup_ch.send(None),
                                _ => {
                                    debug!(
                                        "failure to uv_tcp_init");
                                    let err_data =
                                        uv::ll::get_last_err_data(
                                            loop_ptr);
                                    setup_ch.send(Some(err_data));
                                }
                            }
                        }
                        _ => {
                            debug!("failure to uv_tcp_bind");
                            let err_data = uv::ll::get_last_err_data(
                                loop_ptr);
                            setup_ch.send(Some(err_data));
                        }
                    }
                }
                _ => {
                    debug!("failure to uv_tcp_bind");
                    let err_data = uv::ll::get_last_err_data(
                        loop_ptr);
                    setup_ch.send(Some(err_data));
                }
            }
        }
    }

    let setup_result = setup_po.recv();

    match setup_result {
        Some(ref err_data) => {
            do iotask::interact(iotask) |loop_ptr| {
                unsafe {
                    debug!(
                        "tcp::listen post-kill recv hl interact %?",
                             loop_ptr);
                    *(*server_data_ptr).active = false;
                    uv::ll::close(server_stream_ptr, tcp_lfc_close_cb);
                }
            };
            stream_closed_po.recv();
            match err_data.err_name {
                ~"EACCES" => {
                    debug!("Got EACCES error");
                    result::Err(AccessDenied)
                }
                ~"EADDRINUSE" => {
                    debug!("Got EADDRINUSE error");
                    result::Err(AddressInUse)
                }
                _ => {
                    debug!("Got '%s' '%s' libuv error",
                                    err_data.err_name, err_data.err_msg);
                    result::Err(
                        GenericListenErr(err_data.err_name.clone(),
                                         err_data.err_msg.clone()))
                }
            }
        }
        None => {
            on_establish_cb(kill_ch.clone());
            let kill_result = kill_po.recv();
            do iotask::interact(iotask) |loop_ptr| {
                unsafe {
                    debug!(
                        "tcp::listen post-kill recv hl interact %?",
                             loop_ptr);
                    *(*server_data_ptr).active = false;
                    uv::ll::close(server_stream_ptr, tcp_lfc_close_cb);
                }
            };
            stream_closed_po.recv();
            match kill_result {
                // some failure post bind/listen
                Some(ref err_data) => result::Err(GenericListenErr(
                    err_data.err_name.clone(),
                    err_data.err_msg.clone())),
                // clean exit
                None => result::Ok(())
            }
        }
    }
}


/**
 * Convert a `net::tcp::TcpSocket` to a `net::tcp::TcpSocketBuf`.
 *
 * This function takes ownership of a `net::tcp::TcpSocket`, returning it
 * stored within a buffered wrapper, which can be converted to a `io::Reader`
 * or `io::Writer`
 *
 * # Arguments
 *
 * * `sock` -- a `net::tcp::TcpSocket` that you want to buffer
 *
 * # Returns
 *
 * A buffered wrapper that you can cast as an `io::Reader` or `io::Writer`
 */
pub fn socket_buf(sock: TcpSocket) -> TcpSocketBuf {
    TcpSocketBuf(@mut TcpBufferedSocketData {
        sock: sock, buf: ~[], buf_off: 0
    })
}

/// Convenience methods extending `net::tcp::TcpSocket`
impl TcpSocket {
    pub fn read_start(&self) -> result::Result<@Port<
        result::Result<~[u8], TcpErrData>>, TcpErrData> {
        read_start(self)
    }
    pub fn read_stop(&self) ->
        result::Result<(), TcpErrData> {
        read_stop(self)
    }
    pub fn read(&self, timeout_msecs: uint) ->
        result::Result<~[u8], TcpErrData> {
        read(self, timeout_msecs)
    }
    pub fn read_future(&self, timeout_msecs: uint) ->
        future::Future<result::Result<~[u8], TcpErrData>> {
        read_future(self, timeout_msecs)
    }
    pub fn write(&self, raw_write_data: ~[u8])
        -> result::Result<(), TcpErrData> {
        write(self, raw_write_data)
    }
    pub fn write_future(&self, raw_write_data: ~[u8])
        -> future::Future<result::Result<(), TcpErrData>> {
        write_future(self, raw_write_data)
    }
    pub fn get_peer_addr(&self) -> ip::IpAddr {
        unsafe {
            if self.socket_data.ipv6 {
                let addr = uv::ll::ip6_addr("", 0);
                uv::ll::tcp_getpeername6(self.socket_data.stream_handle_ptr,
                                         &addr);
                ip::Ipv6(addr)
            } else {
                let addr = uv::ll::ip4_addr("", 0);
                uv::ll::tcp_getpeername(self.socket_data.stream_handle_ptr,
                                        &addr);
                ip::Ipv4(addr)
            }
        }
    }
}

/// Implementation of `io::Reader` trait for a buffered `net::tcp::TcpSocket`
impl io::Reader for TcpSocketBuf {
    fn read(&self, buf: &mut [u8], len: uint) -> uint {
        if len == 0 { return 0 }
        let mut count: uint = 0;

        loop {
          assert!(count < len);

          // If possible, copy up to `len` bytes from the internal
          // `data.buf` into `buf`
          let nbuffered = self.data.buf.len() - self.data.buf_off;
          let needed = len - count;
            if nbuffered > 0 {
                unsafe {
                    let ncopy = num::min(nbuffered, needed);
                    let dst = ptr::mut_offset(
                        vec::raw::to_mut_ptr(buf), count);
                    let src = ptr::offset(
                        vec::raw::to_ptr(self.data.buf),
                        self.data.buf_off);
                    ptr::copy_memory(dst, src, ncopy);
                    self.data.buf_off += ncopy;
                    count += ncopy;
                }
          }

          assert!(count <= len);
          if count == len {
              break;
          }

          // We copied all the bytes we had in the internal buffer into
          // the result buffer, but the caller wants more bytes, so we
          // need to read in data from the socket. Note that the internal
          // buffer is of no use anymore as we read all bytes from it,
          // so we can throw it away.
          let read_result = {
            let data = &*self.data;
            read(&data.sock, 0)
          };
          if read_result.is_err() {
              let err_data = read_result.get_err();

              if err_data.err_name == ~"EOF" {
                  *self.end_of_stream = true;
                  break;
              } else {
                  debug!("ERROR sock_buf as io::reader.read err %? %?",
                         err_data.err_name, err_data.err_msg);
                  // As we have already copied data into result buffer,
                  // we cannot simply return 0 here. Instead the error
                  // should show up in a later call to read().
                  break;
              }
          } else {
              self.data.buf = result::unwrap(read_result);
              self.data.buf_off = 0;
          }
        }

        count
    }
    fn read_byte(&self) -> int {
        loop {
          if self.data.buf.len() > self.data.buf_off {
            let c = self.data.buf[self.data.buf_off];
            self.data.buf_off += 1;
            return c as int
          }

          let read_result = {
            let data = &*self.data;
            read(&data.sock, 0)
          };
          if read_result.is_err() {
              let err_data = read_result.get_err();

              if err_data.err_name == ~"EOF" {
                  *self.end_of_stream = true;
                  return -1
              } else {
                  debug!("ERROR sock_buf as io::reader.read err %? %?",
                         err_data.err_name, err_data.err_msg);
                  fail!()
              }
          } else {
              self.data.buf = result::unwrap(read_result);
              self.data.buf_off = 0;
          }
        }
    }
    fn eof(&self) -> bool {
        *self.end_of_stream
    }
    fn seek(&self, dist: int, seek: io::SeekStyle) {
        debug!("tcp_socket_buf seek stub %? %?", dist, seek);
        // noop
    }
    fn tell(&self) -> uint {
        0u // noop
    }
}

/// Implementation of `io::Reader` trait for a buffered `net::tcp::TcpSocket`
impl io::Writer for TcpSocketBuf {
    pub fn write(&self, data: &[u8]) {
        let socket_data_ptr: *TcpSocketData =
            &(*((*(self.data)).sock).socket_data);
        let w_result = write_common_impl(socket_data_ptr,
                                         data.slice(0, data.len()).to_owned());
        if w_result.is_err() {
            let err_data = w_result.get_err();
            debug!(
                "ERROR sock_buf as io::writer.writer err: %? %?",
                err_data.err_name, err_data.err_msg);
        }
    }
    fn seek(&self, dist: int, seek: io::SeekStyle) {
      debug!("tcp_socket_buf seek stub %? %?", dist, seek);
        // noop
    }
    fn tell(&self) -> uint {
        0u
    }
    fn flush(&self) -> int {
        0
    }
    fn get_type(&self) -> io::WriterType {
        io::File
    }
}

// INTERNAL API

fn tear_down_socket_data(socket_data: @TcpSocketData) {
    unsafe {
        let (closed_po, closed_ch) = stream::<()>();
        let closed_ch = SharedChan::new(closed_ch);
        let close_data = TcpSocketCloseData {
            closed_ch: closed_ch
        };
        let close_data_ptr: *TcpSocketCloseData = &close_data;
        let stream_handle_ptr = (*socket_data).stream_handle_ptr;
        do iotask::interact(&(*socket_data).iotask) |loop_ptr| {
            debug!(
                "interact dtor for tcp_socket stream %? loop %?",
                     stream_handle_ptr, loop_ptr);
            uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                           close_data_ptr);
            uv::ll::close(stream_handle_ptr, tcp_socket_dtor_close_cb);
        };
        closed_po.recv();
        //the line below will most likely crash
        //log(debug, fmt!("about to free socket_data at %?", socket_data));
        rustrt::rust_uv_current_kernel_free(stream_handle_ptr
                                            as *libc::c_void);
        debug!("exiting dtor for tcp_socket");
    }
}

// shared implementation for tcp::read
fn read_common_impl(socket_data: *TcpSocketData, timeout_msecs: uint)
    -> result::Result<~[u8],TcpErrData> {
    unsafe {
        use timer;

        debug!("starting tcp::read");
        let iotask = &(*socket_data).iotask;
        let rs_result = read_start_common_impl(socket_data);
        if result::is_err(&rs_result) {
            let err_data = result::get_err(&rs_result);
            result::Err(err_data)
        }
        else {
            debug!("tcp::read before recv_timeout");
            let read_result = if timeout_msecs > 0u {
                timer::recv_timeout(
                    iotask, timeout_msecs, result::unwrap(rs_result))
            } else {
                Some(result::get(&rs_result).recv())
            };
            debug!("tcp::read after recv_timeout");
            match read_result {
                None => {
                    debug!("tcp::read: timed out..");
                    let err_data = TcpErrData {
                        err_name: ~"TIMEOUT",
                        err_msg: ~"req timed out"
                    };
                    read_stop_common_impl(socket_data);
                    result::Err(err_data)
                }
                Some(data_result) => {
                    debug!("tcp::read got data");
                    read_stop_common_impl(socket_data);
                    data_result
                }
            }
        }
    }
}

// shared impl for read_stop
fn read_stop_common_impl(socket_data: *TcpSocketData) ->
    result::Result<(), TcpErrData> {
    unsafe {
        let stream_handle_ptr = (*socket_data).stream_handle_ptr;
        let (stop_po, stop_ch) = stream::<Option<TcpErrData>>();
        do iotask::interact(&(*socket_data).iotask) |loop_ptr| {
            debug!("in interact cb for tcp::read_stop");
            match uv::ll::read_stop(stream_handle_ptr
                                    as *uv::ll::uv_stream_t) {
                0i32 => {
                    debug!("successfully called uv_read_stop");
                    stop_ch.send(None);
                }
                _ => {
                    debug!("failure in calling uv_read_stop");
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    stop_ch.send(Some(err_data.to_tcp_err()));
                }
            }
        }
        match stop_po.recv() {
            Some(err_data) => Err(err_data),
            None => Ok(())
        }
    }
}

// shared impl for read_start
fn read_start_common_impl(socket_data: *TcpSocketData)
    -> result::Result<@Port<
        result::Result<~[u8], TcpErrData>>, TcpErrData> {
    unsafe {
        let stream_handle_ptr = (*socket_data).stream_handle_ptr;
        let (start_po, start_ch) = stream::<Option<uv::ll::uv_err_data>>();
        debug!("in tcp::read_start before interact loop");
        do iotask::interact(&(*socket_data).iotask) |loop_ptr| {
            debug!("in tcp::read_start interact cb %?",
                            loop_ptr);
            match uv::ll::read_start(stream_handle_ptr
                                     as *uv::ll::uv_stream_t,
                                     on_alloc_cb,
                                     on_tcp_read_cb) {
                0i32 => {
                    debug!("success doing uv_read_start");
                    start_ch.send(None);
                }
                _ => {
                    debug!("error attempting uv_read_start");
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    start_ch.send(Some(err_data));
                }
            }
        }
        match start_po.recv() {
            Some(ref err_data) => result::Err(
                err_data.to_tcp_err()),
            None => {
                result::Ok((*socket_data).reader_po)
            }
        }
    }
}

// helper to convert a "class" vector of [u8] to a *[uv::ll::uv_buf_t]

// shared implementation used by write and write_future
fn write_common_impl(socket_data_ptr: *TcpSocketData,
                     raw_write_data: ~[u8])
    -> result::Result<(), TcpErrData> {
    unsafe {
        let write_req_ptr: *uv::ll::uv_write_t =
            &(*socket_data_ptr).write_req;
        let stream_handle_ptr =
            (*socket_data_ptr).stream_handle_ptr;
        let write_buf_vec = ~[
            uv::ll::buf_init(vec::raw::to_ptr(raw_write_data),
                             raw_write_data.len())
        ];
        let write_buf_vec_ptr: *~[uv::ll::uv_buf_t] = &write_buf_vec;
        let (result_po, result_ch) = stream::<TcpWriteResult>();
        let result_ch = SharedChan::new(result_ch);
        let write_data = WriteReqData {
            result_ch: result_ch
        };
        let write_data_ptr: *WriteReqData = &write_data;
        do iotask::interact(&(*socket_data_ptr).iotask) |loop_ptr| {
            debug!("in interact cb for tcp::write %?",
                            loop_ptr);
            match uv::ll::write(write_req_ptr,
                                stream_handle_ptr,
                                write_buf_vec_ptr,
                                tcp_write_complete_cb) {
                0i32 => {
                    debug!("uv_write() invoked successfully");
                    uv::ll::set_data_for_req(write_req_ptr,
                                             write_data_ptr);
                }
                _ => {
                    debug!("error invoking uv_write()");
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    let result_ch = (*write_data_ptr).result_ch.clone();
                    result_ch.send(TcpWriteError(err_data.to_tcp_err()));
                }
            }
        }
        // FIXME (#2656): Instead of passing unsafe pointers to local data,
        // and waiting here for the write to complete, we should transfer
        // ownership of everything to the I/O task and let it deal with the
        // aftermath, so we don't have to sit here blocking.
        match result_po.recv() {
            TcpWriteSuccess => Ok(()),
            TcpWriteError(err_data) => Err(err_data)
        }
    }
}

enum TcpNewConnection {
    NewTcpConn(*uv::ll::uv_tcp_t)
}

struct TcpListenFcData {
    server_stream_ptr: *uv::ll::uv_tcp_t,
    stream_closed_ch: SharedChan<()>,
    kill_ch: SharedChan<Option<TcpErrData>>,
    on_connect_cb: ~fn(*uv::ll::uv_tcp_t),
    iotask: IoTask,
    ipv6: bool,
    active: @mut bool,
}

extern fn tcp_lfc_close_cb(handle: *uv::ll::uv_tcp_t) {
    unsafe {
        let server_data_ptr = uv::ll::get_data_for_uv_handle(
            handle) as *TcpListenFcData;
        let stream_closed_ch = (*server_data_ptr).stream_closed_ch.clone();
        stream_closed_ch.send(());
    }
}

extern fn tcp_lfc_on_connection_cb(handle: *uv::ll::uv_tcp_t,
                                     status: libc::c_int) {
    unsafe {
        let server_data_ptr = uv::ll::get_data_for_uv_handle(handle)
            as *TcpListenFcData;
        let kill_ch = (*server_data_ptr).kill_ch.clone();
        if *(*server_data_ptr).active {
            match status {
              0i32 => ((*server_data_ptr).on_connect_cb)(handle),
              _ => {
                let loop_ptr = uv::ll::get_loop_for_uv_handle(handle);
                kill_ch.send(
                           Some(uv::ll::get_last_err_data(loop_ptr)
                                .to_tcp_err()));
                *(*server_data_ptr).active = false;
              }
            }
        }
    }
}

fn malloc_uv_tcp_t() -> *uv::ll::uv_tcp_t {
    unsafe {
        rustrt::rust_uv_current_kernel_malloc(
            rustrt::rust_uv_helper_uv_tcp_t_size()) as *uv::ll::uv_tcp_t
    }
}

enum TcpConnectResult {
    TcpConnected(TcpSocket),
    TcpConnectError(TcpErrData)
}

enum TcpWriteResult {
    TcpWriteSuccess,
    TcpWriteError(TcpErrData)
}

enum TcpReadStartResult {
    TcpReadStartSuccess(Port<TcpReadResult>),
    TcpReadStartError(TcpErrData)
}

enum TcpReadResult {
    TcpReadData(~[u8]),
    TcpReadDone,
    TcpReadErr(TcpErrData)
}

trait ToTcpErr {
    fn to_tcp_err(&self) -> TcpErrData;
}

impl ToTcpErr for uv::ll::uv_err_data {
    fn to_tcp_err(&self) -> TcpErrData {
        TcpErrData {
            err_name: self.err_name.clone(),
            err_msg: self.err_msg.clone(),
        }
    }
}

extern fn on_tcp_read_cb(stream: *uv::ll::uv_stream_t,
                         nread: libc::ssize_t,
                         buf: uv::ll::uv_buf_t) {
    unsafe {
        debug!("entering on_tcp_read_cb stream: %x nread: %?",
                        stream as uint, nread);
        let loop_ptr = uv::ll::get_loop_for_uv_handle(stream);
        let socket_data_ptr = uv::ll::get_data_for_uv_handle(stream)
            as *TcpSocketData;
        debug!("socket data is %x", socket_data_ptr as uint);
        match nread as int {
          // incoming err.. probably eof
          -1 => {
            let err_data = uv::ll::get_last_err_data(loop_ptr).to_tcp_err();
            debug!("on_tcp_read_cb: incoming err.. name %? msg %?",
                            err_data.err_name, err_data.err_msg);
            let reader_ch = &(*socket_data_ptr).reader_ch;
            reader_ch.send(result::Err(err_data));
          }
          // do nothing .. unneeded buf
          0 => (),
          // have data
          _ => {
            // we have data
            debug!("tcp on_read_cb nread: %d", nread as int);
            let reader_ch = &(*socket_data_ptr).reader_ch;
            let buf_base = uv::ll::get_base_from_buf(buf);
            let new_bytes = vec::from_buf(buf_base, nread as uint);
            reader_ch.send(result::Ok(new_bytes));
          }
        }
        uv::ll::free_base_of_buf(buf);
        debug!("exiting on_tcp_read_cb");
    }
}

extern fn on_alloc_cb(handle: *libc::c_void,
                      suggested_size: size_t)
    -> uv::ll::uv_buf_t {
    unsafe {
        debug!("tcp read on_alloc_cb!");
        let char_ptr = uv::ll::malloc_buf_base_of(suggested_size);
        debug!("tcp read on_alloc_cb h: %? char_ptr: %u sugsize: %u",
                         handle,
                         char_ptr as uint,
                         suggested_size as uint);
        uv::ll::buf_init(char_ptr, suggested_size as uint)
    }
}

struct TcpSocketCloseData {
    closed_ch: SharedChan<()>,
}

extern fn tcp_socket_dtor_close_cb(handle: *uv::ll::uv_tcp_t) {
    unsafe {
        let data = uv::ll::get_data_for_uv_handle(handle)
            as *TcpSocketCloseData;
        let closed_ch = (*data).closed_ch.clone();
        closed_ch.send(());
        debug!("tcp_socket_dtor_close_cb exiting..");
    }
}

extern fn tcp_write_complete_cb(write_req: *uv::ll::uv_write_t,
                              status: libc::c_int) {
    unsafe {
        let write_data_ptr = uv::ll::get_data_for_req(write_req)
            as *WriteReqData;
        if status == 0i32 {
            debug!("successful write complete");
            let result_ch = (*write_data_ptr).result_ch.clone();
            result_ch.send(TcpWriteSuccess);
        } else {
            let stream_handle_ptr = uv::ll::get_stream_handle_from_write_req(
                write_req);
            let loop_ptr = uv::ll::get_loop_for_uv_handle(stream_handle_ptr);
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            debug!("failure to write");
            let result_ch = (*write_data_ptr).result_ch.clone();
            result_ch.send(TcpWriteError(err_data.to_tcp_err()));
        }
    }
}

struct WriteReqData {
    result_ch: SharedChan<TcpWriteResult>,
}

struct ConnectReqData {
    result_ch: SharedChan<ConnAttempt>,
    closed_signal_ch: SharedChan<()>,
}

extern fn stream_error_close_cb(handle: *uv::ll::uv_tcp_t) {
    unsafe {
        let data = uv::ll::get_data_for_uv_handle(handle) as
            *ConnectReqData;
        let closed_signal_ch = (*data).closed_signal_ch.clone();
        closed_signal_ch.send(());
        debug!("exiting steam_error_close_cb for %?", handle);
    }
}

extern fn tcp_connect_close_cb(handle: *uv::ll::uv_tcp_t) {
    debug!("closed client tcp handle %?", handle);
}

extern fn tcp_connect_on_connect_cb(connect_req_ptr: *uv::ll::uv_connect_t,
                                   status: libc::c_int) {
    unsafe {
        let conn_data_ptr = (uv::ll::get_data_for_req(connect_req_ptr)
                          as *ConnectReqData);
        let result_ch = (*conn_data_ptr).result_ch.clone();
        debug!("tcp_connect result_ch %?", result_ch);
        let tcp_stream_ptr =
            uv::ll::get_stream_handle_from_connect_req(connect_req_ptr);
        match status {
          0i32 => {
            debug!("successful tcp connection!");
            result_ch.send(ConnSuccess);
          }
          _ => {
            debug!("error in tcp_connect_on_connect_cb");
            let loop_ptr = uv::ll::get_loop_for_uv_handle(tcp_stream_ptr);
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            debug!("err_data %? %?", err_data.err_name,
                            err_data.err_msg);
            result_ch.send(ConnFailure(err_data));
            uv::ll::set_data_for_uv_handle(tcp_stream_ptr,
                                           conn_data_ptr);
            uv::ll::close(tcp_stream_ptr, stream_error_close_cb);
          }
        }
        debug!("leaving tcp_connect_on_connect_cb");
    }
}

enum ConnAttempt {
    ConnSuccess,
    ConnFailure(uv::ll::uv_err_data)
}

struct TcpSocketData {
    reader_po: @Port<result::Result<~[u8], TcpErrData>>,
    reader_ch: SharedChan<result::Result<~[u8], TcpErrData>>,
    stream_handle_ptr: *uv::ll::uv_tcp_t,
    connect_req: uv::ll::uv_connect_t,
    write_req: uv::ll::uv_write_t,
    ipv6: bool,
    iotask: IoTask,
}

struct TcpBufferedSocketData {
    sock: TcpSocket,
    buf: ~[u8],
    buf_off: uint
}

#[cfg(test)]
mod test {

    use net::ip;
    use net::tcp::{GenericListenErr, TcpConnectErrData, TcpListenErrData};
    use net::tcp::{connect, accept, read, listen, TcpSocket, socket_buf};
    use net;
    use uv::iotask::IoTask;
    use uv;

    use std::cell::Cell;
    use std::comm::{stream, SharedChan};
    use std::io;
    use std::result;
    use std::str;
    use std::task;

    // FIXME don't run on fbsd or linux 32 bit (#2064)
    #[cfg(target_os="win32")]
    #[cfg(target_os="darwin")]
    #[cfg(target_os="linux")]
    #[cfg(target_os="android")]
    mod tcp_ipv4_server_and_client_test {
        #[cfg(target_arch="x86_64")]
        mod impl64 {
            use net::tcp::test::*;

            #[test]
            fn test_gl_tcp_server_and_client_ipv4() {
                impl_gl_tcp_ipv4_server_and_client();
            }
            #[test]
            fn test_gl_tcp_get_peer_addr() {
                impl_gl_tcp_ipv4_get_peer_addr();
            }
            #[test]
            fn test_gl_tcp_ipv4_client_error_connection_refused() {
                impl_gl_tcp_ipv4_client_error_connection_refused();
            }
            #[test]
            fn test_gl_tcp_server_address_in_use() {
                impl_gl_tcp_ipv4_server_address_in_use();
            }
            #[test]
            fn test_gl_tcp_server_access_denied() {
                impl_gl_tcp_ipv4_server_access_denied();
            }
            // Strange failure on Windows. --pcwalton
            #[test]
            #[ignore(cfg(target_os = "win32"))]
            fn test_gl_tcp_ipv4_server_client_reader_writer() {
                impl_gl_tcp_ipv4_server_client_reader_writer();
            }
            #[test]
            fn test_tcp_socket_impl_reader_handles_eof() {
                impl_tcp_socket_impl_reader_handles_eof();
            }
        }
        #[cfg(target_arch="x86")]
        #[cfg(target_arch="arm")]
        #[cfg(target_arch="mips")]
        mod impl32 {
            use net::tcp::test::*;

            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_server_and_client_ipv4() {
                unsafe {
                    impl_gl_tcp_ipv4_server_and_client();
                }
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_get_peer_addr() {
                unsafe {
                    impl_gl_tcp_ipv4_get_peer_addr();
                }
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_ipv4_client_error_connection_refused() {
                unsafe {
                    impl_gl_tcp_ipv4_client_error_connection_refused();
                }
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_server_address_in_use() {
                unsafe {
                    impl_gl_tcp_ipv4_server_address_in_use();
                }
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            #[ignore(cfg(windows), reason = "deadlocking bots")]
            fn test_gl_tcp_server_access_denied() {
                unsafe {
                    impl_gl_tcp_ipv4_server_access_denied();
                }
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            #[ignore(cfg(target_os = "win32"))]
            fn test_gl_tcp_ipv4_server_client_reader_writer() {
                impl_gl_tcp_ipv4_server_client_reader_writer();
            }
        }
    }
    pub fn impl_gl_tcp_ipv4_server_and_client() {
        let hl_loop = &uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 8888u;
        let expected_req = ~"ping";
        let expected_resp = "pong";

        let (server_result_po, server_result_ch) = stream::<~str>();

        let (cont_po, cont_ch) = stream::<()>();
        let cont_ch = SharedChan::new(cont_ch);
        // server
        let hl_loop_clone = hl_loop.clone();
        do task::spawn_sched(task::ManualThreads(1u)) {
            let cont_ch = cont_ch.clone();
            let actual_req = run_tcp_test_server(
                server_ip,
                server_port,
                expected_resp.to_str(),
                cont_ch.clone(),
                &hl_loop_clone);
            server_result_ch.send(actual_req);
        };
        cont_po.recv();
        // client
        debug!("server started, firing up client..");
        let actual_resp_result = run_tcp_test_client(
            server_ip,
            server_port,
            expected_req,
            hl_loop);
        assert!(actual_resp_result.is_ok());
        let actual_resp = actual_resp_result.get();
        let actual_req = server_result_po.recv();
        debug!("REQ: expected: '%s' actual: '%s'",
                       expected_req, actual_req);
        debug!("RESP: expected: '%s' actual: '%s'",
                       expected_resp, actual_resp);
        assert!(actual_req.contains(expected_req));
        assert!(actual_resp.contains(expected_resp));
    }
    pub fn impl_gl_tcp_ipv4_get_peer_addr() {
        let hl_loop = &uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 8887u;
        let expected_resp = "pong";

        let (cont_po, cont_ch) = stream::<()>();
        let cont_ch = SharedChan::new(cont_ch);
        // server
        let hl_loop_clone = hl_loop.clone();
        do task::spawn_sched(task::ManualThreads(1u)) {
            let cont_ch = cont_ch.clone();
            run_tcp_test_server(
                server_ip,
                server_port,
                expected_resp.to_str(),
                cont_ch.clone(),
                &hl_loop_clone);
        };
        cont_po.recv();
        // client
        debug!("server started, firing up client..");
        let server_ip_addr = ip::v4::parse_addr(server_ip);
        let iotask = uv::global_loop::get();
        let connect_result = connect(server_ip_addr, server_port,
                                     &iotask);

        let sock = result::unwrap(connect_result);

        debug!("testing peer address");
        // This is what we are actually testing!
        assert!(net::ip::format_addr(&sock.get_peer_addr()) ==
            ~"127.0.0.1");
        assert_eq!(net::ip::get_port(&sock.get_peer_addr()), 8887);

        // Fulfill the protocol the test server expects
        let resp_bytes = "ping".as_bytes().to_owned();
        tcp_write_single(&sock, resp_bytes);
        debug!("message sent");
        sock.read(0u);
        debug!("result read");
    }
    pub fn impl_gl_tcp_ipv4_client_error_connection_refused() {
        let hl_loop = &uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 8889u;
        let expected_req = ~"ping";
        // client
        debug!("firing up client..");
        let actual_resp_result = run_tcp_test_client(
            server_ip,
            server_port,
            expected_req,
            hl_loop);
        match actual_resp_result.get_err() {
          ConnectionRefused => (),
          _ => fail!("unknown error.. expected connection_refused")
        }
    }
    pub fn impl_gl_tcp_ipv4_server_address_in_use() {
        let hl_loop = &uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 8890u;
        let expected_req = ~"ping";
        let expected_resp = "pong";

        let (cont_po, cont_ch) = stream::<()>();
        let cont_ch = SharedChan::new(cont_ch);
        // server
        let hl_loop_clone = hl_loop.clone();
        do task::spawn_sched(task::ManualThreads(1u)) {
            let cont_ch = cont_ch.clone();
            run_tcp_test_server(
                server_ip,
                server_port,
                expected_resp.to_str(),
                cont_ch.clone(),
                &hl_loop_clone);
        }
        cont_po.recv();
        // this one should fail..
        let listen_err = run_tcp_test_server_fail(
                            server_ip,
                            server_port,
                            hl_loop);
        // client.. just doing this so that the first server tears down
        debug!("server started, firing up client..");
        run_tcp_test_client(
            server_ip,
            server_port,
            expected_req,
            hl_loop);
        match listen_err {
          AddressInUse => {
            assert!(true);
          }
          _ => {
            fail!("expected address_in_use listen error, \
                   but got a different error varient. check logs.");
          }
        }
    }
    pub fn impl_gl_tcp_ipv4_server_access_denied() {
        let hl_loop = &uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 80u;
        // this one should fail..
        let listen_err = run_tcp_test_server_fail(
                            server_ip,
                            server_port,
                            hl_loop);
        match listen_err {
          AccessDenied => {
            assert!(true);
          }
          _ => {
            fail!("expected address_in_use listen error, \
                   but got a different error varient. check logs.");
          }
        }
    }
    pub fn impl_gl_tcp_ipv4_server_client_reader_writer() {

        let iotask = &uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 8891u;
        let expected_req = ~"ping";
        let expected_resp = "pong";

        let (server_result_po, server_result_ch) = stream::<~str>();

        let (cont_po, cont_ch) = stream::<()>();
        let cont_ch = SharedChan::new(cont_ch);
        // server
        let iotask_clone = iotask.clone();
        do task::spawn_sched(task::ManualThreads(1u)) {
            let cont_ch = cont_ch.clone();
            let actual_req = run_tcp_test_server(
                server_ip,
                server_port,
                expected_resp.to_str(),
                cont_ch.clone(),
                &iotask_clone);
            server_result_ch.send(actual_req);
        };
        cont_po.recv();
        // client
        let server_addr = ip::v4::parse_addr(server_ip);
        let conn_result = connect(server_addr, server_port, iotask);
        if result::is_err(&conn_result) {
            assert!(false);
        }
        let sock_buf = @socket_buf(result::unwrap(conn_result));
        buf_write(sock_buf, expected_req);

        // so contrived!
        let actual_resp = buf_read(sock_buf, expected_resp.as_bytes().len());

        let actual_req = server_result_po.recv();
        debug!("REQ: expected: '%s' actual: '%s'",
                       expected_req, actual_req);
        debug!("RESP: expected: '%s' actual: '%s'",
                       expected_resp, actual_resp);
        assert!(actual_req.contains(expected_req));
        assert!(actual_resp.contains(expected_resp));
    }

    pub fn impl_tcp_socket_impl_reader_handles_eof() {
        use std::io::{Reader,ReaderUtil};

        let hl_loop = &uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 10041u;
        let expected_req = ~"GET /";
        let expected_resp = "A string\nwith multiple lines\n";

        let (cont_po, cont_ch) = stream::<()>();
        let cont_ch = SharedChan::new(cont_ch);
        // server
        let hl_loop_clone = hl_loop.clone();
        do task::spawn_sched(task::ManualThreads(1u)) {
            let cont_ch = cont_ch.clone();
            run_tcp_test_server(
                server_ip,
                server_port,
                expected_resp.to_str(),
                cont_ch.clone(),
                &hl_loop_clone);
        };
        cont_po.recv();
        // client
        debug!("server started, firing up client..");
        let server_addr = ip::v4::parse_addr(server_ip);
        let conn_result = connect(server_addr, server_port, hl_loop);
        if result::is_err(&conn_result) {
            assert!(false);
        }
        let sock_buf = @socket_buf(result::unwrap(conn_result));
        buf_write(sock_buf, expected_req);

        let buf_reader = sock_buf as @Reader;
        let actual_response = str::from_bytes(buf_reader.read_whole_stream());
        debug!("Actual response: %s", actual_response);
        assert!(expected_resp == actual_response);
    }

    fn buf_write<W:io::Writer>(w: &W, val: &str) {
        debug!("BUF_WRITE: val len %?", val.len());
        let b_slice = val.as_bytes();
        debug!("BUF_WRITE: b_slice len %?",
               b_slice.len());
        w.write(b_slice)
    }

    fn buf_read<R:io::Reader>(r: &R, len: uint) -> ~str {
        let new_bytes = (*r).read_bytes(len);
        debug!("in buf_read.. new_bytes len: %?",
                        new_bytes.len());
        str::from_bytes(new_bytes)
    }

    fn run_tcp_test_server(server_ip: &str, server_port: uint, resp: ~str,
                          cont_ch: SharedChan<()>,
                          iotask: &IoTask) -> ~str {
        let (server_po, server_ch) = stream::<~str>();
        let server_ch = SharedChan::new(server_ch);
        let server_ip_addr = ip::v4::parse_addr(server_ip);
        let resp_cell = Cell::new(resp);
        let listen_result = listen(server_ip_addr, server_port, 128,
                                   iotask,
            // on_establish_cb -- called when listener is set up
            |kill_ch| {
                debug!("establish_cb %?",
                    kill_ch);
                cont_ch.send(());
            },
            // risky to run this on the loop, but some users
            // will want the POWER
            |new_conn, kill_ch| {
                let resp_cell2 = Cell::new(resp_cell.take());
                debug!("SERVER: new connection!");
                let (cont_po, cont_ch) = stream();
                let server_ch = server_ch.clone();
                do task::spawn_sched(task::ManualThreads(1u)) {
                    debug!("SERVER: starting worker for new req");

                    let accept_result = accept(new_conn);
                    debug!("SERVER: after accept()");
                    if result::is_err(&accept_result) {
                        debug!("SERVER: error accept connection");
                        let err_data = result::get_err(&accept_result);
                        kill_ch.send(Some(err_data));
                        debug!(
                            "SERVER/WORKER: send on err cont ch");
                        cont_ch.send(());
                    }
                    else {
                        debug!("SERVER/WORKER: send on cont ch");
                        cont_ch.send(());
                        let sock = result::unwrap(accept_result);
                        let peer_addr = sock.get_peer_addr();
                        debug!("SERVER: successfully accepted \
                                connection from %s:%u",
                                 ip::format_addr(&peer_addr),
                                 ip::get_port(&peer_addr));
                        let received_req_bytes = read(&sock, 0u);
                        match received_req_bytes {
                          result::Ok(data) => {
                            debug!("SERVER: got REQ str::from_bytes..");
                            debug!("SERVER: REQ data len: %?",
                                            data.len());
                            server_ch.send(
                                str::from_bytes(data));
                            debug!("SERVER: before write");
                            let s = resp_cell2.take();
                            tcp_write_single(&sock, s.as_bytes().to_owned());
                            debug!("SERVER: after write.. die");
                            kill_ch.send(None);
                          }
                          result::Err(err_data) => {
                            debug!("SERVER: error recvd: %s %s",
                                err_data.err_name, err_data.err_msg);
                            kill_ch.send(Some(err_data));
                            server_ch.send(~"");
                          }
                        }
                        debug!("SERVER: worker spinning down");
                    }
                }
                debug!("SERVER: waiting to recv on cont_ch");
                cont_po.recv();
        });
        // err check on listen_result
        if result::is_err(&listen_result) {
            match result::get_err(&listen_result) {
              GenericListenErr(ref name, ref msg) => {
                fail!("SERVER: exited abnormally name %s msg %s", *name, *msg);
              }
              AccessDenied => {
                fail!("SERVER: exited abnormally, got access denied..");
              }
              AddressInUse => {
                fail!("SERVER: exited abnormally, got address in use...");
              }
            }
        }
        let ret_val = server_po.recv();
        debug!("SERVER: exited and got return val: '%s'", ret_val);
        ret_val
    }

    fn run_tcp_test_server_fail(server_ip: &str, server_port: uint,
                                iotask: &IoTask) -> TcpListenErrData {
        let server_ip_addr = ip::v4::parse_addr(server_ip);
        let listen_result = listen(server_ip_addr, server_port, 128,
                                   iotask,
            // on_establish_cb -- called when listener is set up
            |kill_ch| {
                debug!("establish_cb %?", kill_ch);
            },
            |new_conn, kill_ch| {
                fail!("SERVER: shouldn't be called.. %? %?", new_conn, kill_ch);
        });
        // err check on listen_result
        if result::is_err(&listen_result) {
            result::get_err(&listen_result)
        }
        else {
            fail!("SERVER: did not fail as expected")
        }
    }

    fn run_tcp_test_client(server_ip: &str, server_port: uint, resp: &str,
                          iotask: &IoTask) -> result::Result<~str,
                                                    TcpConnectErrData> {
        let server_ip_addr = ip::v4::parse_addr(server_ip);

        debug!("CLIENT: starting..");
        let connect_result = connect(server_ip_addr, server_port,
                                     iotask);
        if result::is_err(&connect_result) {
            debug!("CLIENT: failed to connect");
            let err_data = result::get_err(&connect_result);
            Err(err_data)
        }
        else {
            let sock = result::unwrap(connect_result);
            let resp_bytes = resp.as_bytes().to_owned();
            tcp_write_single(&sock, resp_bytes);
            let read_result = sock.read(0u);
            if read_result.is_err() {
                debug!("CLIENT: failure to read");
                Ok(~"")
            }
            else {
                let ret_val = str::from_bytes(read_result.get());
                debug!("CLIENT: after client_ch recv ret: '%s'",
                   ret_val);
                Ok(ret_val)
            }
        }
    }

    fn tcp_write_single(sock: &TcpSocket, val: ~[u8]) {
        let mut write_result_future = sock.write_future(val);
        let write_result = write_result_future.get();
        if result::is_err(&write_result) {
            debug!("tcp_write_single: write failed!");
            let err_data = result::get_err(&write_result);
            debug!("tcp_write_single err name: %s msg: %s",
                err_data.err_name, err_data.err_msg);
            // meh. torn on what to do here.
            fail!("tcp_write_single failed");
        }
    }
}
