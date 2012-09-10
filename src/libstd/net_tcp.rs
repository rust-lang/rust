//! High-level interface to libuv's TCP functionality

use ip = net_ip;
use uv::iotask;
use uv::iotask::IoTask;
use future_spawn = future::spawn;
// FIXME #1935
// should be able to, but can't atm, replace w/ result::{result, extensions};
use result::*;
use libc::size_t;
use io::{Reader, ReaderUtil, Writer};
use comm = core::comm;

// tcp interfaces
export TcpSocket;
// buffered socket
export TcpSocketBuf, socket_buf;
// errors
export TcpErrData, TcpConnectErrData;
// operations on a tcp_socket
export write, write_future, read_start, read_stop;
// tcp server stuff
export listen, accept;
// tcp client stuff
export connect;

#[nolink]
extern mod rustrt {
    fn rust_uv_current_kernel_malloc(size: libc::c_uint) -> *libc::c_void;
    fn rust_uv_current_kernel_free(mem: *libc::c_void);
    fn rust_uv_helper_uv_tcp_t_size() -> libc::c_uint;
}

/**
 * Encapsulates an open TCP/IP connection through libuv
 *
 * `tcp_socket` is non-copyable/sendable and automagically handles closing the
 * underlying libuv data structures when it goes out of scope. This is the
 * data structure that is used for read/write operations over a TCP stream.
 */
struct TcpSocket {
  socket_data: @TcpSocketData,
  drop {
    unsafe {
        tear_down_socket_data(self.socket_data)
    }
  }
}

fn TcpSocket(socket_data: @TcpSocketData) -> TcpSocket {
    TcpSocket {
        socket_data: socket_data
    }
}

/**
 * A buffered wrapper for `net::tcp::tcp_socket`
 *
 * It is created with a call to `net::tcp::socket_buf()` and has impls that
 * satisfy both the `io::reader` and `io::writer` traits.
 */
struct TcpSocketBuf {
    data: @TcpBufferedSocketData,
}

fn TcpSocketBuf(data: @TcpBufferedSocketData) -> TcpSocketBuf {
    TcpSocketBuf {
        data: data
    }
}

/// Contains raw, string-based, error information returned from libuv
type TcpErrData = {
    err_name: ~str,
    err_msg: ~str
};
/// Details returned as part of a `result::err` result from `tcp::listen`
enum TcpListenErrData {
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
/// Details returned as part of a `result::err` result from `tcp::connect`
enum TcpConnectErrData {
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
 * `net::net::tcp_socket` that can be used to send and receive data to/from
 * the remote host. In the event of failure, a
 * `net::tcp::tcp_connect_err_data` instance will be returned
 */
fn connect(-input_ip: ip::IpAddr, port: uint,
           iotask: IoTask)
    -> result::Result<TcpSocket, TcpConnectErrData> unsafe {
    let result_po = core::comm::Port::<ConnAttempt>();
    let closed_signal_po = core::comm::Port::<()>();
    let conn_data = {
        result_ch: core::comm::Chan(result_po),
        closed_signal_ch: core::comm::Chan(closed_signal_po)
    };
    let conn_data_ptr = ptr::addr_of(conn_data);
    let reader_po = core::comm::Port::<result::Result<~[u8], TcpErrData>>();
    let stream_handle_ptr = malloc_uv_tcp_t();
    *(stream_handle_ptr as *mut uv::ll::uv_tcp_t) = uv::ll::tcp_t();
    let socket_data = @{
        reader_po: reader_po,
        reader_ch: core::comm::Chan(reader_po),
        stream_handle_ptr: stream_handle_ptr,
        connect_req: uv::ll::connect_t(),
        write_req: uv::ll::write_t(),
        iotask: iotask
    };
    let socket_data_ptr = ptr::addr_of(*socket_data);
    log(debug, fmt!("tcp_connect result_ch %?", conn_data.result_ch));
    // get an unsafe representation of our stream_handle_ptr that
    // we can send into the interact cb to be handled in libuv..
    log(debug, fmt!("stream_handle_ptr outside interact %?",
        stream_handle_ptr));
    do iotask::interact(iotask) |loop_ptr| unsafe {
        log(debug, ~"in interact cb for tcp client connect..");
        log(debug, fmt!("stream_handle_ptr in interact %?",
            stream_handle_ptr));
        match uv::ll::tcp_init( loop_ptr, stream_handle_ptr) {
          0i32 => {
            log(debug, ~"tcp_init successful");
            log(debug, ~"dealing w/ ipv4 connection..");
            let connect_req_ptr =
                ptr::addr_of((*socket_data_ptr).connect_req);
            let addr_str = ip::format_addr(&input_ip);
            let connect_result = match input_ip {
              ip::Ipv4(addr) => {
                // have to "recreate" the sockaddr_in/6
                // since the ip_addr discards the port
                // info.. should probably add an additional
                // rust type that actually is closer to
                // what the libuv API expects (ip str + port num)
                log(debug, fmt!("addr: %?", addr));
                let in_addr = uv::ll::ip4_addr(addr_str, port as int);
                uv::ll::tcp_connect(
                    connect_req_ptr,
                    stream_handle_ptr,
                    ptr::addr_of(in_addr),
                    tcp_connect_on_connect_cb)
              }
              ip::Ipv6(addr) => {
                log(debug, fmt!("addr: %?", addr));
                let in_addr = uv::ll::ip6_addr(addr_str, port as int);
                uv::ll::tcp_connect6(
                    connect_req_ptr,
                    stream_handle_ptr,
                    ptr::addr_of(in_addr),
                    tcp_connect_on_connect_cb)
              }
            };
            match connect_result {
              0i32 => {
                log(debug, ~"tcp_connect successful");
                // reusable data that we'll have for the
                // duration..
                uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                           socket_data_ptr as
                                              *libc::c_void);
                // just so the connect_cb can send the
                // outcome..
                uv::ll::set_data_for_req(connect_req_ptr,
                                         conn_data_ptr);
                log(debug, ~"leaving tcp_connect interact cb...");
                // let tcp_connect_on_connect_cb send on
                // the result_ch, now..
              }
              _ => {
                // immediate connect failure.. probably a garbage
                // ip or somesuch
                let err_data = uv::ll::get_last_err_data(loop_ptr);
                core::comm::send((*conn_data_ptr).result_ch,
                           ConnFailure(err_data.to_tcp_err()));
                uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                               conn_data_ptr);
                uv::ll::close(stream_handle_ptr, stream_error_close_cb);
              }
            }
        }
          _ => {
            // failure to create a tcp handle
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            core::comm::send((*conn_data_ptr).result_ch,
                       ConnFailure(err_data.to_tcp_err()));
          }
        }
    };
    match core::comm::recv(result_po) {
      ConnSuccess => {
        log(debug, ~"tcp::connect - received success on result_po");
        result::Ok(TcpSocket(socket_data))
      }
      ConnFailure(err_data) => {
        core::comm::recv(closed_signal_po);
        log(debug, ~"tcp::connect - received failure on result_po");
        // still have to free the malloc'd stream handle..
        rustrt::rust_uv_current_kernel_free(stream_handle_ptr
                                           as *libc::c_void);
        let tcp_conn_err = match err_data.err_name {
          ~"ECONNREFUSED" => ConnectionRefused,
          _ => GenericConnectErr(err_data.err_name, err_data.err_msg)
        };
        result::Err(tcp_conn_err)
      }
    }
}

/**
 * Write binary data to a tcp stream; Blocks until operation completes
 *
 * # Arguments
 *
 * * sock - a `tcp_socket` to write to
 * * raw_write_data - a vector of `~[u8]` that will be written to the stream.
 * This value must remain valid for the duration of the `write` call
 *
 * # Returns
 *
 * A `result` object with a `nil` value as the `ok` variant, or a
 * `tcp_err_data` value as the `err` variant
 */
fn write(sock: TcpSocket, raw_write_data: ~[u8])
    -> result::Result<(), TcpErrData> unsafe {
    let socket_data_ptr = ptr::addr_of(*(sock.socket_data));
    write_common_impl(socket_data_ptr, raw_write_data)
}

/**
 * Write binary data to tcp stream; Returns a `future::future` value
 * immediately
 *
 * # Safety
 *
 * This function can produce unsafe results if:
 *
 * 1. the call to `write_future` is made
 * 2. the `future::future` value returned is never resolved via
 * `future::get`
 * 3. and then the `tcp_socket` passed in to `write_future` leaves
 * scope and is destructed before the task that runs the libuv write
 * operation completes.
 *
 * As such: If using `write_future`, always be sure to resolve the returned
 * `future` so as to ensure libuv doesn't try to access a released write
 * handle. Otherwise, use the blocking `tcp::write` function instead.
 *
 * # Arguments
 *
 * * sock - a `tcp_socket` to write to
 * * raw_write_data - a vector of `~[u8]` that will be written to the stream.
 * This value must remain valid for the duration of the `write` call
 *
 * # Returns
 *
 * A `future` value that, once the `write` operation completes, resolves to a
 * `result` object with a `nil` value as the `ok` variant, or a `tcp_err_data`
 * value as the `err` variant
 */
fn write_future(sock: TcpSocket, raw_write_data: ~[u8])
    -> future::Future<result::Result<(), TcpErrData>> unsafe {
    let socket_data_ptr = ptr::addr_of(*(sock.socket_data));
    do future_spawn {
        let data_copy = copy(raw_write_data);
        write_common_impl(socket_data_ptr, data_copy)
    }
}

/**
 * Begin reading binary data from an open TCP connection; used with
 * `read_stop`
 *
 * # Arguments
 *
 * * sock -- a `net::tcp::tcp_socket` for the connection to read from
 *
 * # Returns
 *
 * * A `result` instance that will either contain a
 * `core::comm::port<tcp_read_result>` that the user can read (and
 * optionally, loop on) from until `read_stop` is called, or a
 * `tcp_err_data` record
 */
fn read_start(sock: TcpSocket)
    -> result::Result<comm::Port<
        result::Result<~[u8], TcpErrData>>, TcpErrData> unsafe {
    let socket_data = ptr::addr_of(*(sock.socket_data));
    read_start_common_impl(socket_data)
}

/**
 * Stop reading from an open TCP connection; used with `read_start`
 *
 * # Arguments
 *
 * * `sock` - a `net::tcp::tcp_socket` that you wish to stop reading on
 */
fn read_stop(sock: TcpSocket,
             -read_port: comm::Port<result::Result<~[u8], TcpErrData>>) ->
    result::Result<(), TcpErrData> unsafe {
    log(debug, fmt!("taking the read_port out of commission %?", read_port));
    let socket_data = ptr::addr_of(*sock.socket_data);
    read_stop_common_impl(socket_data)
}

/**
 * Reads a single chunk of data from `tcp_socket`; block until data/error
 * recv'd
 *
 * Does a blocking read operation for a single chunk of data from a
 * `tcp_socket` until a data arrives or an error is received. The provided
 * `timeout_msecs` value is used to raise an error if the timeout period
 * passes without any data received.
 *
 * # Arguments
 *
 * * `sock` - a `net::tcp::tcp_socket` that you wish to read from
 * * `timeout_msecs` - a `uint` value, in msecs, to wait before dropping the
 * read attempt. Pass `0u` to wait indefinitely
 */
fn read(sock: TcpSocket, timeout_msecs: uint)
    -> result::Result<~[u8],TcpErrData> {
    let socket_data = ptr::addr_of(*(sock.socket_data));
    read_common_impl(socket_data, timeout_msecs)
}

/**
 * Reads a single chunk of data; returns a `future::future<~[u8]>`
 * immediately
 *
 * Does a non-blocking read operation for a single chunk of data from a
 * `tcp_socket` and immediately returns a `future` value representing the
 * result. When resolving the returned `future`, it will block until data
 * arrives or an error is received. The provided `timeout_msecs`
 * value is used to raise an error if the timeout period passes without any
 * data received.
 *
 * # Safety
 *
 * This function can produce unsafe results if the call to `read_future` is
 * made, the `future::future` value returned is never resolved via
 * `future::get`, and then the `tcp_socket` passed in to `read_future` leaves
 * scope and is destructed before the task that runs the libuv read
 * operation completes.
 *
 * As such: If using `read_future`, always be sure to resolve the returned
 * `future` so as to ensure libuv doesn't try to access a released read
 * handle. Otherwise, use the blocking `tcp::read` function instead.
 *
 * # Arguments
 *
 * * `sock` - a `net::tcp::tcp_socket` that you wish to read from
 * * `timeout_msecs` - a `uint` value, in msecs, to wait before dropping the
 * read attempt. Pass `0u` to wait indefinitely
 */
fn read_future(sock: TcpSocket, timeout_msecs: uint)
    -> future::Future<result::Result<~[u8],TcpErrData>> {
    let socket_data = ptr::addr_of(*(sock.socket_data));
    do future_spawn {
        read_common_impl(socket_data, timeout_msecs)
    }
}

/**
 * Bind an incoming client connection to a `net::tcp::tcp_socket`
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
 * `net::tcp::tcp_socket` is completed.
 *
 * # Example
 *
 * Here, the `new_conn` is used in conjunction with `accept` from within
 * a task spawned by the `new_connect_cb` passed into `listen`
 *
 * ~~~~~~~~~~~
 * net::tcp::listen(remote_ip, remote_port, backlog)
 *     // this callback is ran once after the connection is successfully
 *     // set up
 *     {|kill_ch|
 *       // pass the kill_ch to your main loop or wherever you want
 *       // to be able to externally kill the server from
 *     }
 *     // this callback is ran when a new connection arrives
 *     {|new_conn, kill_ch|
 *     let cont_po = core::comm::port::<option<tcp_err_data>>();
 *     let cont_ch = core::comm::chan(cont_po);
 *     task::spawn {||
 *         let accept_result = net::tcp::accept(new_conn);
 *         if accept_result.is_err() {
 *             core::comm::send(cont_ch, result::get_err(accept_result));
 *             // fail?
 *         }
 *         else {
 *             let sock = result::get(accept_result);
 *             core::comm::send(cont_ch, true);
 *             // do work here
 *         }
 *     };
 *     match core::comm::recv(cont_po) {
 *       // shut down listen()
 *       some(err_data) { core::comm::send(kill_chan, some(err_data)) }
 *       // wait for next connection
 *       none {}
 *     }
 * };
 * ~~~~~~~~~~~
 *
 * # Arguments
 *
 * * `new_conn` - an opaque value used to create a new `tcp_socket`
 *
 * # Returns
 *
 * On success, this function will return a `net::tcp::tcp_socket` as the
 * `ok` variant of a `result`. The `net::tcp::tcp_socket` is anchored within
 * the task that `accept` was called within for its lifetime. On failure,
 * this function will return a `net::tcp::tcp_err_data` record
 * as the `err` variant of a `result`.
 */
fn accept(new_conn: TcpNewConnection)
    -> result::Result<TcpSocket, TcpErrData> unsafe {

    match new_conn{
      NewTcpConn(server_handle_ptr) => {
        let server_data_ptr = uv::ll::get_data_for_uv_handle(
            server_handle_ptr) as *TcpListenFcData;
        let reader_po = core::comm::Port();
        let iotask = (*server_data_ptr).iotask;
        let stream_handle_ptr = malloc_uv_tcp_t();
        *(stream_handle_ptr as *mut uv::ll::uv_tcp_t) = uv::ll::tcp_t();
        let client_socket_data = @{
            reader_po: reader_po,
            reader_ch: core::comm::Chan(reader_po),
            stream_handle_ptr : stream_handle_ptr,
            connect_req : uv::ll::connect_t(),
            write_req : uv::ll::write_t(),
            iotask : iotask
        };
        let client_socket_data_ptr = ptr::addr_of(*client_socket_data);
        let client_stream_handle_ptr =
            (*client_socket_data_ptr).stream_handle_ptr;

        let result_po = core::comm::Port::<Option<TcpErrData>>();
        let result_ch = core::comm::Chan(result_po);

        // UNSAFE LIBUV INTERACTION BEGIN
        // .. normally this happens within the context of
        // a call to uv::hl::interact.. but we're breaking
        // the rules here because this always has to be
        // called within the context of a listen() new_connect_cb
        // callback (or it will likely fail and drown your cat)
        log(debug, ~"in interact cb for tcp::accept");
        let loop_ptr = uv::ll::get_loop_for_uv_handle(
            server_handle_ptr);
        match uv::ll::tcp_init(loop_ptr, client_stream_handle_ptr) {
          0i32 => {
            log(debug, ~"uv_tcp_init successful for client stream");
            match uv::ll::accept(
                server_handle_ptr as *libc::c_void,
                client_stream_handle_ptr as *libc::c_void) {
              0i32 => {
                log(debug, ~"successfully accepted client connection");
                uv::ll::set_data_for_uv_handle(client_stream_handle_ptr,
                                               client_socket_data_ptr
                                                   as *libc::c_void);
                core::comm::send(result_ch, None);
              }
              _ => {
                log(debug, ~"failed to accept client conn");
                core::comm::send(result_ch, Some(
                    uv::ll::get_last_err_data(loop_ptr).to_tcp_err()));
              }
            }
          }
          _ => {
            log(debug, ~"failed to init client stream");
            core::comm::send(result_ch, Some(
                uv::ll::get_last_err_data(loop_ptr).to_tcp_err()));
          }
        }
        // UNSAFE LIBUV INTERACTION END
        match core::comm::recv(result_po) {
          Some(err_data) => result::Err(err_data),
          None => result::Ok(TcpSocket(client_socket_data))
        }
      }
    }
}

/**
 * Bind to a given IP/port and listen for new connections
 *
 * # Arguments
 *
 * * `host_ip` - a `net::ip::ip_addr` representing a unique IP
 * (versions 4 or 6)
 * * `port` - a uint representing the port to listen on
 * * `backlog` - a uint representing the number of incoming connections
 * to cache in memory
 * * `hl_loop` - a `uv::hl::high_level_loop` that the tcp request will run on
 * * `on_establish_cb` - a callback that is evaluated if/when the listener
 * is successfully established. it takes no parameters
 * * `new_connect_cb` - a callback to be evaluated, on the libuv thread,
 * whenever a client attempts to conect on the provided ip/port. the
 * callback's arguments are:
 *     * `new_conn` - an opaque type that can be passed to
 *     `net::tcp::accept` in order to be converted to a `tcp_socket`.
 *     * `kill_ch` - channel of type `core::comm::chan<option<tcp_err_data>>`.
 *     this channel can be used to send a message to cause `listen` to begin
 *     closing the underlying libuv data structures.
 *
 * # returns
 *
 * a `result` instance containing empty data of type `()` on a
 * successful/normal shutdown, and a `tcp_listen_err_data` enum in the event
 * of listen exiting because of an error
 */
fn listen(-host_ip: ip::IpAddr, port: uint, backlog: uint,
          iotask: IoTask,
          on_establish_cb: fn~(comm::Chan<Option<TcpErrData>>),
          +new_connect_cb: fn~(TcpNewConnection,
                               comm::Chan<Option<TcpErrData>>))
    -> result::Result<(), TcpListenErrData> unsafe {
    do listen_common(host_ip, port, backlog, iotask, on_establish_cb)
        // on_connect_cb
        |handle| unsafe {
            let server_data_ptr = uv::ll::get_data_for_uv_handle(handle)
                as *TcpListenFcData;
            let new_conn = NewTcpConn(handle);
            let kill_ch = (*server_data_ptr).kill_ch;
            new_connect_cb(new_conn, kill_ch);
    }
}

fn listen_common(-host_ip: ip::IpAddr, port: uint, backlog: uint,
          iotask: IoTask,
          on_establish_cb: fn~(comm::Chan<Option<TcpErrData>>),
          -on_connect_cb: fn~(*uv::ll::uv_tcp_t))
    -> result::Result<(), TcpListenErrData> unsafe {
    let stream_closed_po = core::comm::Port::<()>();
    let kill_po = core::comm::Port::<Option<TcpErrData>>();
    let kill_ch = core::comm::Chan(kill_po);
    let server_stream = uv::ll::tcp_t();
    let server_stream_ptr = ptr::addr_of(server_stream);
    let server_data = {
        server_stream_ptr: server_stream_ptr,
        stream_closed_ch: core::comm::Chan(stream_closed_po),
        kill_ch: kill_ch,
        on_connect_cb: on_connect_cb,
        iotask: iotask,
        mut active: true
    };
    let server_data_ptr = ptr::addr_of(server_data);

    let setup_result = do core::comm::listen |setup_ch| {
        // this is to address a compiler warning about
        // an implicit copy.. it seems that double nested
        // will defeat a move sigil, as is done to the host_ip
        // arg above.. this same pattern works w/o complaint in
        // tcp::connect (because the iotask::interact cb isn't
        // nested within a core::comm::listen block)
        let loc_ip = copy(host_ip);
        do iotask::interact(iotask) |loop_ptr| unsafe {
            match uv::ll::tcp_init(loop_ptr, server_stream_ptr) {
              0i32 => {
                uv::ll::set_data_for_uv_handle(
                    server_stream_ptr,
                    server_data_ptr);
                let addr_str = ip::format_addr(&loc_ip);
                let bind_result = match loc_ip {
                  ip::Ipv4(addr) => {
                    log(debug, fmt!("addr: %?", addr));
                    let in_addr = uv::ll::ip4_addr(addr_str, port as int);
                    uv::ll::tcp_bind(server_stream_ptr,
                                     ptr::addr_of(in_addr))
                  }
                  ip::Ipv6(addr) => {
                    log(debug, fmt!("addr: %?", addr));
                    let in_addr = uv::ll::ip6_addr(addr_str, port as int);
                    uv::ll::tcp_bind6(server_stream_ptr,
                                     ptr::addr_of(in_addr))
                  }
                };
                match bind_result {
                  0i32 => {
                    match uv::ll::listen(server_stream_ptr,
                                       backlog as libc::c_int,
                                       tcp_lfc_on_connection_cb) {
                      0i32 => core::comm::send(setup_ch, None),
                      _ => {
                        log(debug, ~"failure to uv_listen()");
                        let err_data = uv::ll::get_last_err_data(loop_ptr);
                        core::comm::send(setup_ch, Some(err_data));
                      }
                    }
                  }
                  _ => {
                    log(debug, ~"failure to uv_tcp_bind");
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    core::comm::send(setup_ch, Some(err_data));
                  }
                }
              }
              _ => {
                log(debug, ~"failure to uv_tcp_init");
                let err_data = uv::ll::get_last_err_data(loop_ptr);
                core::comm::send(setup_ch, Some(err_data));
              }
            }
        };
        setup_ch.recv()
    };
    match setup_result {
      Some(err_data) => {
        do iotask::interact(iotask) |loop_ptr| unsafe {
            log(debug, fmt!("tcp::listen post-kill recv hl interact %?",
                            loop_ptr));
            (*server_data_ptr).active = false;
            uv::ll::close(server_stream_ptr, tcp_lfc_close_cb);
        };
        stream_closed_po.recv();
        match err_data.err_name {
          ~"EACCES" => {
            log(debug, ~"Got EACCES error");
            result::Err(AccessDenied)
          }
          ~"EADDRINUSE" => {
            log(debug, ~"Got EADDRINUSE error");
            result::Err(AddressInUse)
          }
          _ => {
            log(debug, fmt!("Got '%s' '%s' libuv error",
                            err_data.err_name, err_data.err_msg));
            result::Err(
                GenericListenErr(err_data.err_name, err_data.err_msg))
          }
        }
      }
      None => {
        on_establish_cb(kill_ch);
        let kill_result = core::comm::recv(kill_po);
        do iotask::interact(iotask) |loop_ptr| unsafe {
            log(debug, fmt!("tcp::listen post-kill recv hl interact %?",
                            loop_ptr));
            (*server_data_ptr).active = false;
            uv::ll::close(server_stream_ptr, tcp_lfc_close_cb);
        };
        stream_closed_po.recv();
        match kill_result {
          // some failure post bind/listen
          Some(err_data) => result::Err(GenericListenErr(err_data.err_name,
                                                           err_data.err_msg)),
          // clean exit
          None => result::Ok(())
        }
      }
    }
}

/**
 * Convert a `net::tcp::tcp_socket` to a `net::tcp::tcp_socket_buf`.
 *
 * This function takes ownership of a `net::tcp::tcp_socket`, returning it
 * stored within a buffered wrapper, which can be converted to a `io::reader`
 * or `io::writer`
 *
 * # Arguments
 *
 * * `sock` -- a `net::tcp::tcp_socket` that you want to buffer
 *
 * # Returns
 *
 * A buffered wrapper that you can cast as an `io::reader` or `io::writer`
 */
fn socket_buf(-sock: TcpSocket) -> TcpSocketBuf {
    TcpSocketBuf(@{ sock: sock, mut buf: ~[] })
}

/// Convenience methods extending `net::tcp::tcp_socket`
impl TcpSocket {
    fn read_start() -> result::Result<comm::Port<
        result::Result<~[u8], TcpErrData>>, TcpErrData> {
        read_start(self)
    }
    fn read_stop(-read_port:
                 comm::Port<result::Result<~[u8], TcpErrData>>) ->
        result::Result<(), TcpErrData> {
        read_stop(self, read_port)
    }
    fn read(timeout_msecs: uint) ->
        result::Result<~[u8], TcpErrData> {
        read(self, timeout_msecs)
    }
    fn read_future(timeout_msecs: uint) ->
        future::Future<result::Result<~[u8], TcpErrData>> {
        read_future(self, timeout_msecs)
    }
    fn write(raw_write_data: ~[u8])
        -> result::Result<(), TcpErrData> {
        write(self, raw_write_data)
    }
    fn write_future(raw_write_data: ~[u8])
        -> future::Future<result::Result<(), TcpErrData>> {
        write_future(self, raw_write_data)
    }
}

/// Implementation of `io::reader` trait for a buffered `net::tcp::tcp_socket`
impl TcpSocketBuf: io::Reader {
    fn read(buf: &[mut u8], len: uint) -> uint {
        // Loop until our buffer has enough data in it for us to read from.
        while self.data.buf.len() < len {
            let read_result = read(self.data.sock, 0u);
            if read_result.is_err() {
                let err_data = read_result.get_err();

                if err_data.err_name == ~"EOF" {
                    break;
                } else {
                    debug!("ERROR sock_buf as io::reader.read err %? %?",
                           err_data.err_name, err_data.err_msg);

                    return 0;
                }
            }
            else {
                vec::push_all(self.data.buf, result::unwrap(read_result));
            }
        }

        let count = uint::min(len, self.data.buf.len());

        let mut data = ~[];
        self.data.buf <-> data;

        vec::u8::memcpy(buf, vec::view(data, 0, data.len()), count);

        vec::push_all(self.data.buf, vec::view(data, count, data.len()));

        count
    }
    fn read_byte() -> int {
        let bytes = ~[0];
        if self.read(bytes, 1u) == 0 { fail } else { bytes[0] as int }
    }
    fn unread_byte(amt: int) {
        vec::unshift((*(self.data)).buf, amt as u8);
    }
    fn eof() -> bool {
        false // noop
    }
    fn seek(dist: int, seek: io::SeekStyle) {
        log(debug, fmt!("tcp_socket_buf seek stub %? %?", dist, seek));
        // noop
    }
    fn tell() -> uint {
        0u // noop
    }
}

/// Implementation of `io::reader` trait for a buffered `net::tcp::tcp_socket`
impl TcpSocketBuf: io::Writer {
    fn write(data: &[const u8]) unsafe {
        let socket_data_ptr =
            ptr::addr_of(*((*(self.data)).sock).socket_data);
        let w_result = write_common_impl(socket_data_ptr,
                                        vec::slice(data, 0, vec::len(data)));
        if w_result.is_err() {
            let err_data = w_result.get_err();
            log(debug, fmt!("ERROR sock_buf as io::writer.writer err: %? %?",
                             err_data.err_name, err_data.err_msg));
        }
    }
    fn seek(dist: int, seek: io::SeekStyle) {
      log(debug, fmt!("tcp_socket_buf seek stub %? %?", dist, seek));
        // noop
    }
    fn tell() -> uint {
        0u
    }
    fn flush() -> int {
        0
    }
    fn get_type() -> io::WriterType {
        io::File
    }
}

// INTERNAL API

fn tear_down_socket_data(socket_data: @TcpSocketData) unsafe {
    let closed_po = core::comm::Port::<()>();
    let closed_ch = core::comm::Chan(closed_po);
    let close_data = {
        closed_ch: closed_ch
    };
    let close_data_ptr = ptr::addr_of(close_data);
    let stream_handle_ptr = (*socket_data).stream_handle_ptr;
    do iotask::interact((*socket_data).iotask) |loop_ptr| unsafe {
        log(debug, fmt!("interact dtor for tcp_socket stream %? loop %?",
            stream_handle_ptr, loop_ptr));
        uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                       close_data_ptr);
        uv::ll::close(stream_handle_ptr, tcp_socket_dtor_close_cb);
    };
    core::comm::recv(closed_po);
    log(debug, fmt!("about to free socket_data at %?", socket_data));
    rustrt::rust_uv_current_kernel_free(stream_handle_ptr
                                       as *libc::c_void);
    log(debug, ~"exiting dtor for tcp_socket");
}

// shared implementation for tcp::read
fn read_common_impl(socket_data: *TcpSocketData, timeout_msecs: uint)
    -> result::Result<~[u8],TcpErrData> unsafe {
    log(debug, ~"starting tcp::read");
    let iotask = (*socket_data).iotask;
    let rs_result = read_start_common_impl(socket_data);
    if result::is_err(rs_result) {
        let err_data = result::get_err(rs_result);
        result::Err(err_data)
    }
    else {
        log(debug, ~"tcp::read before recv_timeout");
        let read_result = if timeout_msecs > 0u {
            timer::recv_timeout(
               iotask, timeout_msecs, result::get(rs_result))
        } else {
            Some(core::comm::recv(result::get(rs_result)))
        };
        log(debug, ~"tcp::read after recv_timeout");
        match read_result {
          None => {
            log(debug, ~"tcp::read: timed out..");
            let err_data = {
                err_name: ~"TIMEOUT",
                err_msg: ~"req timed out"
            };
            read_stop_common_impl(socket_data);
            result::Err(err_data)
          }
          Some(data_result) => {
            log(debug, ~"tcp::read got data");
            read_stop_common_impl(socket_data);
            data_result
          }
        }
    }
}

// shared impl for read_stop
fn read_stop_common_impl(socket_data: *TcpSocketData) ->
    result::Result<(), TcpErrData> unsafe {
    let stream_handle_ptr = (*socket_data).stream_handle_ptr;
    let stop_po = core::comm::Port::<Option<TcpErrData>>();
    let stop_ch = core::comm::Chan(stop_po);
    do iotask::interact((*socket_data).iotask) |loop_ptr| unsafe {
        log(debug, ~"in interact cb for tcp::read_stop");
        match uv::ll::read_stop(stream_handle_ptr as *uv::ll::uv_stream_t) {
          0i32 => {
            log(debug, ~"successfully called uv_read_stop");
            core::comm::send(stop_ch, None);
          }
          _ => {
            log(debug, ~"failure in calling uv_read_stop");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            core::comm::send(stop_ch, Some(err_data.to_tcp_err()));
          }
        }
    };
    match core::comm::recv(stop_po) {
      Some(err_data) => result::Err(err_data.to_tcp_err()),
      None => result::Ok(())
    }
}

// shared impl for read_start
fn read_start_common_impl(socket_data: *TcpSocketData)
    -> result::Result<comm::Port<
        result::Result<~[u8], TcpErrData>>, TcpErrData> unsafe {
    let stream_handle_ptr = (*socket_data).stream_handle_ptr;
    let start_po = core::comm::Port::<Option<uv::ll::uv_err_data>>();
    let start_ch = core::comm::Chan(start_po);
    log(debug, ~"in tcp::read_start before interact loop");
    do iotask::interact((*socket_data).iotask) |loop_ptr| unsafe {
        log(debug, fmt!("in tcp::read_start interact cb %?", loop_ptr));
        match uv::ll::read_start(stream_handle_ptr as *uv::ll::uv_stream_t,
                               on_alloc_cb,
                               on_tcp_read_cb) {
          0i32 => {
            log(debug, ~"success doing uv_read_start");
            core::comm::send(start_ch, None);
          }
          _ => {
            log(debug, ~"error attempting uv_read_start");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            core::comm::send(start_ch, Some(err_data));
          }
        }
    };
    match core::comm::recv(start_po) {
      Some(err_data) => result::Err(err_data.to_tcp_err()),
      None => result::Ok((*socket_data).reader_po)
    }
}

// helper to convert a "class" vector of [u8] to a *[uv::ll::uv_buf_t]

// shared implementation used by write and write_future
fn write_common_impl(socket_data_ptr: *TcpSocketData,
                     raw_write_data: ~[u8])
    -> result::Result<(), TcpErrData> unsafe {
    let write_req_ptr = ptr::addr_of((*socket_data_ptr).write_req);
    let stream_handle_ptr =
        (*socket_data_ptr).stream_handle_ptr;
    let write_buf_vec =  ~[ uv::ll::buf_init(
        vec::unsafe::to_ptr(raw_write_data),
        vec::len(raw_write_data)) ];
    let write_buf_vec_ptr = ptr::addr_of(write_buf_vec);
    let result_po = core::comm::Port::<TcpWriteResult>();
    let write_data = {
        result_ch: core::comm::Chan(result_po)
    };
    let write_data_ptr = ptr::addr_of(write_data);
    do iotask::interact((*socket_data_ptr).iotask) |loop_ptr| unsafe {
        log(debug, fmt!("in interact cb for tcp::write %?", loop_ptr));
        match uv::ll::write(write_req_ptr,
                          stream_handle_ptr,
                          write_buf_vec_ptr,
                          tcp_write_complete_cb) {
          0i32 => {
            log(debug, ~"uv_write() invoked successfully");
            uv::ll::set_data_for_req(write_req_ptr, write_data_ptr);
          }
          _ => {
            log(debug, ~"error invoking uv_write()");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            core::comm::send((*write_data_ptr).result_ch,
                       TcpWriteError(err_data.to_tcp_err()));
          }
        }
    };
    // FIXME (#2656): Instead of passing unsafe pointers to local data,
    // and waiting here for the write to complete, we should transfer
    // ownership of everything to the I/O task and let it deal with the
    // aftermath, so we don't have to sit here blocking.
    match core::comm::recv(result_po) {
      TcpWriteSuccess => result::Ok(()),
      TcpWriteError(err_data) => result::Err(err_data.to_tcp_err())
    }
}

enum TcpNewConnection {
    NewTcpConn(*uv::ll::uv_tcp_t)
}

type TcpListenFcData = {
    server_stream_ptr: *uv::ll::uv_tcp_t,
    stream_closed_ch: comm::Chan<()>,
    kill_ch: comm::Chan<Option<TcpErrData>>,
    on_connect_cb: fn~(*uv::ll::uv_tcp_t),
    iotask: IoTask,
    mut active: bool
};

extern fn tcp_lfc_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let server_data_ptr = uv::ll::get_data_for_uv_handle(
        handle) as *TcpListenFcData;
    core::comm::send((*server_data_ptr).stream_closed_ch, ());
}

extern fn tcp_lfc_on_connection_cb(handle: *uv::ll::uv_tcp_t,
                                     status: libc::c_int) unsafe {
    let server_data_ptr = uv::ll::get_data_for_uv_handle(handle)
        as *TcpListenFcData;
    let kill_ch = (*server_data_ptr).kill_ch;
    if (*server_data_ptr).active {
        match status {
          0i32 => (*server_data_ptr).on_connect_cb(handle),
          _ => {
            let loop_ptr = uv::ll::get_loop_for_uv_handle(handle);
            core::comm::send(kill_ch,
                       Some(uv::ll::get_last_err_data(loop_ptr)
                            .to_tcp_err()));
            (*server_data_ptr).active = false;
          }
        }
    }
}

fn malloc_uv_tcp_t() -> *uv::ll::uv_tcp_t unsafe {
    rustrt::rust_uv_current_kernel_malloc(
        rustrt::rust_uv_helper_uv_tcp_t_size()) as *uv::ll::uv_tcp_t
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
    TcpReadStartSuccess(comm::Port<TcpReadResult>),
    TcpReadStartError(TcpErrData)
}

enum TcpReadResult {
    TcpReadData(~[u8]),
    TcpReadDone,
    TcpReadErr(TcpErrData)
}

trait ToTcpErr {
    fn to_tcp_err() -> TcpErrData;
}

impl uv::ll::uv_err_data: ToTcpErr {
    fn to_tcp_err() -> TcpErrData {
        { err_name: self.err_name, err_msg: self.err_msg }
    }
}

extern fn on_tcp_read_cb(stream: *uv::ll::uv_stream_t,
                    nread: libc::ssize_t,
                    ++buf: uv::ll::uv_buf_t) unsafe {
    log(debug, fmt!("entering on_tcp_read_cb stream: %? nread: %?",
                    stream, nread));
    let loop_ptr = uv::ll::get_loop_for_uv_handle(stream);
    let socket_data_ptr = uv::ll::get_data_for_uv_handle(stream)
        as *TcpSocketData;
    match nread as int {
      // incoming err.. probably eof
      -1 => {
        let err_data = uv::ll::get_last_err_data(loop_ptr).to_tcp_err();
        log(debug, fmt!("on_tcp_read_cb: incoming err.. name %? msg %?",
                        err_data.err_name, err_data.err_msg));
        let reader_ch = (*socket_data_ptr).reader_ch;
        core::comm::send(reader_ch, result::Err(err_data));
      }
      // do nothing .. unneeded buf
      0 => (),
      // have data
      _ => {
        // we have data
        log(debug, fmt!("tcp on_read_cb nread: %d", nread as int));
        let reader_ch = (*socket_data_ptr).reader_ch;
        let buf_base = uv::ll::get_base_from_buf(buf);
        let new_bytes = vec::unsafe::from_buf(buf_base, nread as uint);
        core::comm::send(reader_ch, result::Ok(new_bytes));
      }
    }
    uv::ll::free_base_of_buf(buf);
    log(debug, ~"exiting on_tcp_read_cb");
}

extern fn on_alloc_cb(handle: *libc::c_void,
                     ++suggested_size: size_t)
    -> uv::ll::uv_buf_t unsafe {
    log(debug, ~"tcp read on_alloc_cb!");
    let char_ptr = uv::ll::malloc_buf_base_of(suggested_size);
    log(debug, fmt!("tcp read on_alloc_cb h: %? char_ptr: %u sugsize: %u",
                     handle,
                     char_ptr as uint,
                     suggested_size as uint));
    uv::ll::buf_init(char_ptr, suggested_size as uint)
}

type TcpSocketCloseData = {
    closed_ch: comm::Chan<()>
};

extern fn tcp_socket_dtor_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let data = uv::ll::get_data_for_uv_handle(handle)
        as *TcpSocketCloseData;
    let closed_ch = (*data).closed_ch;
    core::comm::send(closed_ch, ());
    log(debug, ~"tcp_socket_dtor_close_cb exiting..");
}

extern fn tcp_write_complete_cb(write_req: *uv::ll::uv_write_t,
                              status: libc::c_int) unsafe {
    let write_data_ptr = uv::ll::get_data_for_req(write_req)
        as *WriteReqData;
    if status == 0i32 {
        log(debug, ~"successful write complete");
        core::comm::send((*write_data_ptr).result_ch, TcpWriteSuccess);
    } else {
        let stream_handle_ptr = uv::ll::get_stream_handle_from_write_req(
            write_req);
        let loop_ptr = uv::ll::get_loop_for_uv_handle(stream_handle_ptr);
        let err_data = uv::ll::get_last_err_data(loop_ptr);
        log(debug, ~"failure to write");
        core::comm::send((*write_data_ptr).result_ch,
                         TcpWriteError(err_data));
    }
}

type WriteReqData = {
    result_ch: comm::Chan<TcpWriteResult>
};

type ConnectReqData = {
    result_ch: comm::Chan<ConnAttempt>,
    closed_signal_ch: comm::Chan<()>
};

extern fn stream_error_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let data = uv::ll::get_data_for_uv_handle(handle) as
        *ConnectReqData;
    core::comm::send((*data).closed_signal_ch, ());
    log(debug, fmt!("exiting steam_error_close_cb for %?", handle));
}

extern fn tcp_connect_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    log(debug, fmt!("closed client tcp handle %?", handle));
}

extern fn tcp_connect_on_connect_cb(connect_req_ptr: *uv::ll::uv_connect_t,
                                   status: libc::c_int) unsafe {
    let conn_data_ptr = (uv::ll::get_data_for_req(connect_req_ptr)
                      as *ConnectReqData);
    let result_ch = (*conn_data_ptr).result_ch;
    log(debug, fmt!("tcp_connect result_ch %?", result_ch));
    let tcp_stream_ptr =
        uv::ll::get_stream_handle_from_connect_req(connect_req_ptr);
    match status {
      0i32 => {
        log(debug, ~"successful tcp connection!");
        core::comm::send(result_ch, ConnSuccess);
      }
      _ => {
        log(debug, ~"error in tcp_connect_on_connect_cb");
        let loop_ptr = uv::ll::get_loop_for_uv_handle(tcp_stream_ptr);
        let err_data = uv::ll::get_last_err_data(loop_ptr);
        log(debug, fmt!("err_data %? %?", err_data.err_name,
                        err_data.err_msg));
        core::comm::send(result_ch, ConnFailure(err_data));
        uv::ll::set_data_for_uv_handle(tcp_stream_ptr,
                                       conn_data_ptr);
        uv::ll::close(tcp_stream_ptr, stream_error_close_cb);
      }
    }
    log(debug, ~"leaving tcp_connect_on_connect_cb");
}

enum ConnAttempt {
    ConnSuccess,
    ConnFailure(uv::ll::uv_err_data)
}

type TcpSocketData = {
    reader_po: comm::Port<result::Result<~[u8], TcpErrData>>,
    reader_ch: comm::Chan<result::Result<~[u8], TcpErrData>>,
    stream_handle_ptr: *uv::ll::uv_tcp_t,
    connect_req: uv::ll::uv_connect_t,
    write_req: uv::ll::uv_write_t,
    iotask: IoTask
};

type TcpBufferedSocketData = {
    sock: TcpSocket,
    mut buf: ~[u8]
};

//#[cfg(test)]
mod test {
    // FIXME don't run on fbsd or linux 32 bit (#2064)
    #[cfg(target_os="win32")]
    #[cfg(target_os="darwin")]
    #[cfg(target_os="linux")]
    mod tcp_ipv4_server_and_client_test {
        #[cfg(target_arch="x86_64")]
        mod impl64 {
            #[test]
            fn test_gl_tcp_server_and_client_ipv4() unsafe {
                impl_gl_tcp_ipv4_server_and_client();
            }
            #[test]
            fn test_gl_tcp_ipv4_client_error_connection_refused() unsafe {
                impl_gl_tcp_ipv4_client_error_connection_refused();
            }
            #[test]
            fn test_gl_tcp_server_address_in_use() unsafe {
                impl_gl_tcp_ipv4_server_address_in_use();
            }
            #[test]
            fn test_gl_tcp_server_access_denied() unsafe {
                impl_gl_tcp_ipv4_server_access_denied();
            }
            #[test]
            fn test_gl_tcp_ipv4_server_client_reader_writer() {
                impl_gl_tcp_ipv4_server_client_reader_writer();
            }

        }
        #[cfg(target_arch="x86")]
        mod impl32 {
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_server_and_client_ipv4() unsafe {
                impl_gl_tcp_ipv4_server_and_client();
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_ipv4_client_error_connection_refused() unsafe {
                impl_gl_tcp_ipv4_client_error_connection_refused();
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_server_address_in_use() unsafe {
                impl_gl_tcp_ipv4_server_address_in_use();
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            #[ignore(cfg(windows), reason = "deadlocking bots")]
            fn test_gl_tcp_server_access_denied() unsafe {
                impl_gl_tcp_ipv4_server_access_denied();
            }
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_ipv4_server_client_reader_writer() {
                impl_gl_tcp_ipv4_server_client_reader_writer();
            }
        }
    }
    fn impl_gl_tcp_ipv4_server_and_client() {
        let hl_loop = uv::global_loop::get();
        let server_ip = ~"127.0.0.1";
        let server_port = 8888u;
        let expected_req = ~"ping";
        let expected_resp = ~"pong";

        let server_result_po = core::comm::Port::<~str>();
        let server_result_ch = core::comm::Chan(server_result_po);

        let cont_po = core::comm::Port::<()>();
        let cont_ch = core::comm::Chan(cont_po);
        // server
        do task::spawn_sched(task::ManualThreads(1u)) {
            let actual_req = do comm::listen |server_ch| {
                run_tcp_test_server(
                    server_ip,
                    server_port,
                    expected_resp,
                    server_ch,
                    cont_ch,
                    hl_loop)
            };
            server_result_ch.send(actual_req);
        };
        core::comm::recv(cont_po);
        // client
        log(debug, ~"server started, firing up client..");
        let actual_resp_result = do core::comm::listen |client_ch| {
            run_tcp_test_client(
                server_ip,
                server_port,
                expected_req,
                client_ch,
                hl_loop)
        };
        assert actual_resp_result.is_ok();
        let actual_resp = actual_resp_result.get();
        let actual_req = core::comm::recv(server_result_po);
        log(debug, fmt!("REQ: expected: '%s' actual: '%s'",
                       expected_req, actual_req));
        log(debug, fmt!("RESP: expected: '%s' actual: '%s'",
                       expected_resp, actual_resp));
        assert str::contains(actual_req, expected_req);
        assert str::contains(actual_resp, expected_resp);
    }
    fn impl_gl_tcp_ipv4_client_error_connection_refused() {
        let hl_loop = uv::global_loop::get();
        let server_ip = ~"127.0.0.1";
        let server_port = 8889u;
        let expected_req = ~"ping";
        // client
        log(debug, ~"firing up client..");
        let actual_resp_result = do core::comm::listen |client_ch| {
            run_tcp_test_client(
                server_ip,
                server_port,
                expected_req,
                client_ch,
                hl_loop)
        };
        match actual_resp_result.get_err() {
          ConnectionRefused => (),
          _ => fail ~"unknown error.. expected connection_refused"
        }
    }
    fn impl_gl_tcp_ipv4_server_address_in_use() {
        let hl_loop = uv::global_loop::get();
        let server_ip = ~"127.0.0.1";
        let server_port = 8890u;
        let expected_req = ~"ping";
        let expected_resp = ~"pong";

        let server_result_po = core::comm::Port::<~str>();
        let server_result_ch = core::comm::Chan(server_result_po);

        let cont_po = core::comm::Port::<()>();
        let cont_ch = core::comm::Chan(cont_po);
        // server
        do task::spawn_sched(task::ManualThreads(1u)) {
            let actual_req = do comm::listen |server_ch| {
                run_tcp_test_server(
                    server_ip,
                    server_port,
                    expected_resp,
                    server_ch,
                    cont_ch,
                    hl_loop)
            };
            server_result_ch.send(actual_req);
        };
        core::comm::recv(cont_po);
        // this one should fail..
        let listen_err = run_tcp_test_server_fail(
                            server_ip,
                            server_port,
                            hl_loop);
        // client.. just doing this so that the first server tears down
        log(debug, ~"server started, firing up client..");
        do core::comm::listen |client_ch| {
            run_tcp_test_client(
                server_ip,
                server_port,
                expected_req,
                client_ch,
                hl_loop)
        };
        match listen_err {
          AddressInUse => {
            assert true;
          }
          _ => {
            fail ~"expected address_in_use listen error,"+
                ~"but got a different error varient. check logs.";
          }
        }
    }
    fn impl_gl_tcp_ipv4_server_access_denied() {
        let hl_loop = uv::global_loop::get();
        let server_ip = ~"127.0.0.1";
        let server_port = 80u;
        // this one should fail..
        let listen_err = run_tcp_test_server_fail(
                            server_ip,
                            server_port,
                            hl_loop);
        match listen_err {
          AccessDenied => {
            assert true;
          }
          _ => {
            fail ~"expected address_in_use listen error,"+
                      ~"but got a different error varient. check logs.";
          }
        }
    }
    fn impl_gl_tcp_ipv4_server_client_reader_writer() {
        /*
         XXX: Causes an ICE.

        let iotask = uv::global_loop::get();
        let server_ip = ~"127.0.0.1";
        let server_port = 8891u;
        let expected_req = ~"ping";
        let expected_resp = ~"pong";

        let server_result_po = core::comm::port::<~str>();
        let server_result_ch = core::comm::chan(server_result_po);

        let cont_po = core::comm::port::<()>();
        let cont_ch = core::comm::chan(cont_po);
        // server
        do task::spawn_sched(task::ManualThreads(1u)) {
            let actual_req = do comm::listen |server_ch| {
                run_tcp_test_server(
                    server_ip,
                    server_port,
                    expected_resp,
                    server_ch,
                    cont_ch,
                    iotask)
            };
            server_result_ch.send(actual_req);
        };
        core::comm::recv(cont_po);
        // client
        let server_addr = ip::v4::parse_addr(server_ip);
        let conn_result = connect(server_addr, server_port, iotask);
        if result::is_err(conn_result) {
            assert false;
        }
        let sock_buf = @socket_buf(result::unwrap(conn_result));
        buf_write(sock_buf, expected_req);

        // so contrived!
        let actual_resp = do str::as_bytes(expected_resp) |resp_buf| {
            buf_read(sock_buf, vec::len(resp_buf))
        };

        let actual_req = core::comm::recv(server_result_po);
        log(debug, fmt!("REQ: expected: '%s' actual: '%s'",
                       expected_req, actual_req));
        log(debug, fmt!("RESP: expected: '%s' actual: '%s'",
                       expected_resp, actual_resp));
        assert str::contains(actual_req, expected_req);
        assert str::contains(actual_resp, expected_resp);
        */
    }

    fn buf_write<W:io::Writer>(+w: &W, val: ~str) {
        log(debug, fmt!("BUF_WRITE: val len %?", str::len(val)));
        do str::byte_slice(val) |b_slice| {
            log(debug, fmt!("BUF_WRITE: b_slice len %?",
                            vec::len(b_slice)));
            w.write(b_slice)
        }
    }

    fn buf_read<R:io::Reader>(+r: &R, len: uint) -> ~str {
        let new_bytes = (*r).read_bytes(len);
        log(debug, fmt!("in buf_read.. new_bytes len: %?",
                        vec::len(new_bytes)));
        str::from_bytes(new_bytes)
    }

    fn run_tcp_test_server(server_ip: ~str, server_port: uint, resp: ~str,
                          server_ch: comm::Chan<~str>,
                          cont_ch: comm::Chan<()>,
                          iotask: IoTask) -> ~str {
        let server_ip_addr = ip::v4::parse_addr(server_ip);
        let listen_result = listen(server_ip_addr, server_port, 128u, iotask,
            // on_establish_cb -- called when listener is set up
            |kill_ch| {
                log(debug, fmt!("establish_cb %?",
                    kill_ch));
                core::comm::send(cont_ch, ());
            },
            // risky to run this on the loop, but some users
            // will want the POWER
            |new_conn, kill_ch| {
            log(debug, ~"SERVER: new connection!");
            do comm::listen |cont_ch| {
                do task::spawn_sched(task::ManualThreads(1u)) {
                    log(debug, ~"SERVER: starting worker for new req");

                    let accept_result = accept(new_conn);
                    log(debug, ~"SERVER: after accept()");
                    if result::is_err(accept_result) {
                        log(debug, ~"SERVER: error accept connection");
                        let err_data = result::get_err(accept_result);
                        core::comm::send(kill_ch, Some(err_data));
                        log(debug,
                            ~"SERVER/WORKER: send on err cont ch");
                        cont_ch.send(());
                    }
                    else {
                        log(debug,
                            ~"SERVER/WORKER: send on cont ch");
                        cont_ch.send(());
                        let sock = result::unwrap(accept_result);
                        log(debug, ~"SERVER: successfully accepted"+
                            ~"connection!");
                        let received_req_bytes = read(sock, 0u);
                        match received_req_bytes {
                          result::Ok(data) => {
                            log(debug, ~"SERVER: got REQ str::from_bytes..");
                            log(debug, fmt!("SERVER: REQ data len: %?",
                                            vec::len(data)));
                            server_ch.send(
                                str::from_bytes(data));
                            log(debug, ~"SERVER: before write");
                            tcp_write_single(sock, str::to_bytes(resp));
                            log(debug, ~"SERVER: after write.. die");
                            core::comm::send(kill_ch, None);
                          }
                          result::Err(err_data) => {
                            log(debug, fmt!("SERVER: error recvd: %s %s",
                                err_data.err_name, err_data.err_msg));
                            core::comm::send(kill_ch, Some(err_data));
                            server_ch.send(~"");
                          }
                        }
                        log(debug, ~"SERVER: worker spinning down");
                    }
                }
                log(debug, ~"SERVER: waiting to recv on cont_ch");
                cont_ch.recv()
            };
            log(debug, ~"SERVER: recv'd on cont_ch..leaving listen cb");
        });
        // err check on listen_result
        if result::is_err(listen_result) {
            match result::get_err(listen_result) {
              GenericListenErr(name, msg) => {
                fail fmt!("SERVER: exited abnormally name %s msg %s",
                                name, msg);
              }
              AccessDenied => {
                fail ~"SERVER: exited abnormally, got access denied..";
              }
              AddressInUse => {
                fail ~"SERVER: exited abnormally, got address in use...";
              }
            }
        }
        let ret_val = server_ch.recv();
        log(debug, fmt!("SERVER: exited and got return val: '%s'", ret_val));
        ret_val
    }

    fn run_tcp_test_server_fail(server_ip: ~str, server_port: uint,
                          iotask: IoTask) -> TcpListenErrData {
        let server_ip_addr = ip::v4::parse_addr(server_ip);
        let listen_result = listen(server_ip_addr, server_port, 128u, iotask,
            // on_establish_cb -- called when listener is set up
            |kill_ch| {
                log(debug, fmt!("establish_cb %?",
                    kill_ch));
            },
            |new_conn, kill_ch| {
                fail fmt!("SERVER: shouldn't be called.. %? %?",
                           new_conn, kill_ch);
        });
        // err check on listen_result
        if result::is_err(listen_result) {
            result::get_err(listen_result)
        }
        else {
            fail ~"SERVER: did not fail as expected"
        }
    }

    fn run_tcp_test_client(server_ip: ~str, server_port: uint, resp: ~str,
                          client_ch: comm::Chan<~str>,
                          iotask: IoTask) -> result::Result<~str,
                                                    TcpConnectErrData> {
        let server_ip_addr = ip::v4::parse_addr(server_ip);

        log(debug, ~"CLIENT: starting..");
        let connect_result = connect(server_ip_addr, server_port, iotask);
        if result::is_err(connect_result) {
            log(debug, ~"CLIENT: failed to connect");
            let err_data = result::get_err(connect_result);
            Err(err_data)
        }
        else {
            let sock = result::unwrap(connect_result);
            let resp_bytes = str::to_bytes(resp);
            tcp_write_single(sock, resp_bytes);
            let read_result = sock.read(0u);
            if read_result.is_err() {
                log(debug, ~"CLIENT: failure to read");
                Ok(~"")
            }
            else {
                client_ch.send(str::from_bytes(read_result.get()));
                let ret_val = client_ch.recv();
                log(debug, fmt!("CLIENT: after client_ch recv ret: '%s'",
                   ret_val));
                Ok(ret_val)
            }
        }
    }

    fn tcp_write_single(sock: TcpSocket, val: ~[u8]) {
        let write_result_future = sock.write_future(val);
        let write_result = write_result_future.get();
        if result::is_err(write_result) {
            log(debug, ~"tcp_write_single: write failed!");
            let err_data = result::get_err(write_result);
            log(debug, fmt!("tcp_write_single err name: %s msg: %s",
                err_data.err_name, err_data.err_msg));
            // meh. torn on what to do here.
            fail ~"tcp_write_single failed";
        }
    }
}
