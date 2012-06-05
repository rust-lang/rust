#[doc="
High-level interface to libuv's TCP functionality
"];

// FIXME: Fewer import *'s
import ip = net_ip;
import uv::iotask;
import uv::iotask::iotask;
import comm::*;
import result::*;
import str::*;
import future::*;
import libc::size_t;

// data
export tcp_socket, tcp_conn_port, tcp_err_data;
// operations on a tcp_socket
export write, write_future, read_start, read_stop;
// tcp server stuff
export listen_for_conn, accept;
export new_listener, conn_recv, conn_recv_spawn, conn_peek;
// tcp client stuff
export connect;
// helper methods
export conn_port_methods, sock_methods;

#[nolink]
native mod rustrt {
    fn rust_uv_current_kernel_malloc(size: libc::c_uint) -> *libc::c_void;
    fn rust_uv_current_kernel_free(mem: *libc::c_void);
    fn rust_uv_helper_uv_tcp_t_size() -> libc::c_uint;
}

#[doc="
Encapsulates an open TCP/IP connection through libuv

`tcp_socket` is non-copyable/sendable and automagically handles closing the
underlying libuv data structures when it goes out of scope. This is the
data structure that is used for read/write operations over a TCP stream.
"]
resource tcp_socket(socket_data: @tcp_socket_data)
    unsafe {
    let closed_po = comm::port::<()>();
    let closed_ch = comm::chan(closed_po);
    let close_data = {
        closed_ch: closed_ch
    };
    let close_data_ptr = ptr::addr_of(close_data);
    let stream_handle_ptr = (*socket_data).stream_handle_ptr;
    iotask::interact((*socket_data).iotask) {|loop_ptr|
        log(debug, #fmt("interact dtor for tcp_socket stream %? loop %?",
            stream_handle_ptr, loop_ptr));
        uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                       close_data_ptr);
        uv::ll::close(stream_handle_ptr, tcp_socket_dtor_close_cb);
    };
    comm::recv(closed_po);
    log(debug, #fmt("about to free socket_data at %?", socket_data));
    rustrt::rust_uv_current_kernel_free(stream_handle_ptr
                                       as *libc::c_void);
    log(debug, "exiting dtor for tcp_socket");
}

resource tcp_conn_port(conn_data: @tcp_conn_port_data) unsafe {
    let conn_data_ptr = ptr::addr_of(*conn_data);
    let server_stream_ptr = ptr::addr_of((*conn_data_ptr).server_stream);
    let stream_closed_po = (*conn_data).stream_closed_po;
    let iotask = (*conn_data_ptr).iotask;
    iotask::interact(iotask) {|loop_ptr|
        log(debug, #fmt("dtor for tcp_conn_port loop: %?",
                       loop_ptr));
        uv::ll::close(server_stream_ptr, tcp_nl_close_cb);
    }
    comm::recv(stream_closed_po);
}

#[doc="
Contains raw, string-based, error information returned from libuv
"]
type tcp_err_data = {
    err_name: str,
    err_msg: str
};

#[doc="
Initiate a client connection over TCP/IP

# Arguments

* `ip` - The IP address (versions 4 or 6) of the remote host
* `port` - the unsigned integer of the desired remote host port
* `iotask` - a `uv::iotask` that the tcp request will run on

# Returns

A `result` that, if the operation succeeds, contains a `tcp_socket` that
can be used to send and receive data to/from the remote host. In the event
of failure, a `tcp_err_data` will be returned
"]
fn connect(input_ip: ip::ip_addr, port: uint,
           iotask: iotask)
    -> result::result<tcp_socket, tcp_err_data> unsafe {
    let result_po = comm::port::<conn_attempt>();
    let closed_signal_po = comm::port::<()>();
    let conn_data = {
        result_ch: comm::chan(result_po),
        closed_signal_ch: comm::chan(closed_signal_po)
    };
    let conn_data_ptr = ptr::addr_of(conn_data);
    let reader_po = comm::port::<result::result<[u8], tcp_err_data>>();
    let stream_handle_ptr = malloc_uv_tcp_t();
    *(stream_handle_ptr as *mut uv::ll::uv_tcp_t) = uv::ll::tcp_t();
    let socket_data = @{
        reader_po: reader_po,
        reader_ch: comm::chan(reader_po),
        stream_handle_ptr: stream_handle_ptr,
        connect_req: uv::ll::connect_t(),
        write_req: uv::ll::write_t(),
        iotask: iotask
    };
    let socket_data_ptr = ptr::addr_of(*socket_data);
    log(debug, #fmt("tcp_connect result_ch %?", conn_data.result_ch));
    // get an unsafe representation of our stream_handle_ptr that
    // we can send into the interact cb to be handled in libuv..
    log(debug, #fmt("stream_handle_ptr outside interact %?",
        stream_handle_ptr));
    iotask::interact(iotask) {|loop_ptr|
        log(debug, "in interact cb for tcp client connect..");
        log(debug, #fmt("stream_handle_ptr in interact %?",
            stream_handle_ptr));
        alt uv::ll::tcp_init( loop_ptr, stream_handle_ptr) {
          0i32 {
            log(debug, "tcp_init successful");
            alt input_ip {
              ipv4 {
                log(debug, "dealing w/ ipv4 connection..");
                let tcp_addr = ipv4_ip_addr_to_sockaddr_in(input_ip,
                                                           port);
                let tcp_addr_ptr = ptr::addr_of(tcp_addr);
                let connect_req_ptr =
                    ptr::addr_of((*socket_data_ptr).connect_req);
                alt uv::ll::tcp_connect(
                    connect_req_ptr,
                    stream_handle_ptr,
                    tcp_addr_ptr,
                    tcp_connect_on_connect_cb) {
                  0i32 {
                    log(debug, "tcp_connect successful");
                    // reusable data that we'll have for the
                    // duration..
                    uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                               socket_data_ptr as
                                                  *libc::c_void);
                    // just so the connect_cb can send the
                    // outcome..
                    uv::ll::set_data_for_req(connect_req_ptr,
                                             conn_data_ptr);
                    log(debug, "leaving tcp_connect interact cb...");
                    // let tcp_connect_on_connect_cb send on
                    // the result_ch, now..
                  }
                  _ {
                    // immediate connect failure.. probably a garbage
                    // ip or somesuch
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    comm::send((*conn_data_ptr).result_ch,
                               conn_failure(err_data.to_tcp_err()));
                    uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                                   conn_data_ptr);
                    uv::ll::close(stream_handle_ptr, stream_error_close_cb);
                  }
                }
              }
            }
        }
          _ {
            // failure to create a tcp handle
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            comm::send((*conn_data_ptr).result_ch,
                       conn_failure(err_data.to_tcp_err()));
          }
        }
    };
    alt comm::recv(result_po) {
      conn_success {
        log(debug, "tcp::connect - received success on result_po");
        result::ok(tcp_socket(socket_data))
      }
      conn_failure(err_data) {
        comm::recv(closed_signal_po);
        log(debug, "tcp::connect - received failure on result_po");
        result::err(err_data.to_tcp_err())
      }
    }
}

#[doc="
Write binary data to a tcp stream; Blocks until operation completes

# Arguments

* sock - a `tcp_socket` to write to
* raw_write_data - a vector of `[u8]` that will be written to the stream.
This value must remain valid for the duration of the `write` call

# Returns

A `result` object with a `nil` value as the `ok` variant, or a `tcp_err_data`
value as the `err` variant
"]
fn write(sock: tcp_socket, raw_write_data: [u8])
    -> result::result<(), tcp_err_data> unsafe {
    let socket_data_ptr = ptr::addr_of(**sock);
    write_common_impl(socket_data_ptr, raw_write_data)
}

#[doc="
Write binary data to tcp stream; Returns a `future::future` value immediately

# Safety

This function can produce unsafe results if the call to `write_future` is
made, the `future::future` value returned is never resolved via
`future::get`, and then the `tcp_socket` passed in to `write_future` leaves
scope and is destructed before the task that runs the libuv write
operation completes.

As such: If using `write_future`, always be sure to resolve the returned
`future` so as to ensure libuv doesn't try to access a released write handle.
Otherwise, use the blocking `tcp::write` function instead.

# Arguments

* sock - a `tcp_socket` to write to
* raw_write_data - a vector of `[u8]` that will be written to the stream.
This value must remain valid for the duration of the `write` call

# Returns

A `future` value that, once the `write` operation completes, resolves to a
`result` object with a `nil` value as the `ok` variant, or a `tcp_err_data`
value as the `err` variant
"]
fn write_future(sock: tcp_socket, raw_write_data: [u8])
    -> future::future<result::result<(), tcp_err_data>> unsafe {
    let socket_data_ptr = ptr::addr_of(**sock);
    future::spawn {||
        write_common_impl(socket_data_ptr, raw_write_data)
    }
}

#[doc="
Begin reading binary data from an open TCP connection; used with `read_stop`

# Arguments

* sock -- a `net::tcp::tcp_socket` for the connection to read from

# Returns

* A `result` instance that will either contain a
`comm::port<tcp_read_result>` that the user can read (and optionally, loop
on) from until `read_stop` is called, or a `tcp_err_data` record
"]
fn read_start(sock: tcp_socket)
    -> result::result<comm::port<
        result::result<[u8], tcp_err_data>>, tcp_err_data> unsafe {
    let socket_data = ptr::addr_of(**sock);
    read_start_common_impl(socket_data)
}

#[doc="
Stop reading from an open TCP connection; used with `read_start`

# Arguments

* `sock` - a `net::tcp::tcp_socket` that you wish to stop reading on
"]
fn read_stop(sock: tcp_socket) ->
    result::result<(), tcp_err_data> unsafe {
    let socket_data = ptr::addr_of(**sock);
    read_stop_common_impl(socket_data)
}

#[doc="
Reads a single chunk of data from `tcp_socket`; block until data/error recv'd

Does a blocking read operation for a single chunk of data from a `tcp_socket`
until a data arrives or an error is received. The provided `timeout_msecs`
value is used to raise an error if the timeout period passes without any
data received.

# Arguments

* `sock` - a `net::tcp::tcp_socket` that you wish to read from
* `timeout_msecs` - a `uint` value, in msecs, to wait before dropping the
read attempt. Pass `0u` to wait indefinitely
"]
fn read(sock: tcp_socket, timeout_msecs: uint)
    -> result::result<[u8],tcp_err_data> {
    let socket_data = ptr::addr_of(**sock);
    read_common_impl(socket_data, timeout_msecs)
}

#[doc="
Reads a single chunk of data; returns a `future::future<[u8]>` immediately

Does a non-blocking read operation for a single chunk of data from a
`tcp_socket` and immediately returns a `future` value representing the
result. When resolving the returned `future`, it will block until data
arrives or an error is received. The provided `timeout_msecs`
value is used to raise an error if the timeout period passes without any
data received.

# Safety

This function can produce unsafe results if the call to `read_future` is
made, the `future::future` value returned is never resolved via
`future::get`, and then the `tcp_socket` passed in to `read_future` leaves
scope and is destructed before the task that runs the libuv read
operation completes.

As such: If using `read_future`, always be sure to resolve the returned
`future` so as to ensure libuv doesn't try to access a released read handle.
Otherwise, use the blocking `tcp::read` function instead.

# Arguments

* `sock` - a `net::tcp::tcp_socket` that you wish to read from
* `timeout_msecs` - a `uint` value, in msecs, to wait before dropping the
read attempt. Pass `0u` to wait indefinitely
"]
fn read_future(sock: tcp_socket, timeout_msecs: uint)
    -> future::future<result::result<[u8],tcp_err_data>> {
    let socket_data = ptr::addr_of(**sock);
    future::spawn {||
        read_common_impl(socket_data, timeout_msecs)
    }
}

#[doc="
Bind to a given IP/port and listen for new connections

# Arguments

* `host_ip` - a `net::ip::ip_addr` representing a unique IP
(versions 4 or 6)
* `port` - a uint representing the port to listen on
* `backlog` - a uint representing the number of incoming connections
to cache in memory
* `hl_loop` - a `uv::hl::high_level_loop` that the tcp request will run on

# Returns

A `result` instance containing either a `tcp_conn_port` which can used
to listen for, and accept, new connections, or a `tcp_err_data` if
failure to create the tcp listener occurs
"]
fn new_listener(host_ip: ip::ip_addr, port: uint, backlog: uint,
                iotask: iotask)
    -> result::result<tcp_conn_port, tcp_err_data> unsafe {
    let stream_closed_po = comm::port::<()>();
    let stream_closed_ch = comm::chan(stream_closed_po);
    let new_conn_po = comm::port::<result::result<*uv::ll::uv_tcp_t,
                                                  tcp_err_data>>();
    let new_conn_ch = comm::chan(new_conn_po);
    // FIXME: This shared box should not be captured in the i/o task
    // Make it a unique pointer.
    let server_data: @tcp_conn_port_data = @{
        server_stream: uv::ll::tcp_t(),
        stream_closed_po: stream_closed_po,
        stream_closed_ch: stream_closed_ch,
        iotask: iotask,
        new_conn_po: new_conn_po,
        new_conn_ch: new_conn_ch
    };
    let server_data_ptr = ptr::addr_of(*server_data);
    let server_stream_ptr = ptr::addr_of((*server_data_ptr)
                                         .server_stream);

    let setup_po = comm::port::<option<tcp_err_data>>();
    let setup_ch = comm::chan(setup_po);
    iotask::interact(iotask) {|loop_ptr|
        let tcp_addr = ipv4_ip_addr_to_sockaddr_in(host_ip,
                                                   port);
        alt uv::ll::tcp_init(loop_ptr, server_stream_ptr) {
          0i32 {
            alt uv::ll::tcp_bind(server_stream_ptr,
                                 ptr::addr_of(tcp_addr)) {
              0i32 {
                alt uv::ll::listen(server_stream_ptr,
                                   backlog as libc::c_int,
                                   tcp_nl_on_connection_cb) {
                  0i32 {
                    uv::ll::set_data_for_uv_handle(
                        server_stream_ptr,
                        server_data_ptr);
                    comm::send(setup_ch, none);
                  }
                  _ {
                    log(debug, "failure to uv_listen()");
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    comm::send(setup_ch, some(err_data));
                  }
                }
              }
              _ {
                log(debug, "failure to uv_tcp_bind");
                let err_data = uv::ll::get_last_err_data(loop_ptr);
                comm::send(setup_ch, some(err_data));
              }
            }
          }
          _ {
            log(debug, "failure to uv_tcp_init");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            comm::send(setup_ch, some(err_data));
          }
        }
    };
    alt comm::recv(setup_po) {
      some(err_data) {
        // we failed to bind/list w/ libuv
        result::err(err_data.to_tcp_err())
      }
      none {
        result::ok(tcp_conn_port(server_data))
      }
    }
}

#[doc="
Block on a `net::tcp::tcp_conn_port` until a new connection arrives

This function behaves similarly to `comm::recv()`

# Arguments

* server_port -- a `net::tcp::tcp_conn_port` that you wish to listen
on for an incoming connection

# Returns

A `result` object containing a `net::tcp::tcp_socket`, ready for immediate
use, as the `ok` varient, or a `net::tcp::tcp_err_data` for the `err`
variant
"]
fn conn_recv(server_port: tcp_conn_port)
    -> result::result<tcp_socket, tcp_err_data> {
    let new_conn_po = (**server_port).new_conn_po;
    let iotask = (**server_port).iotask;
    let new_conn_result = comm::recv(new_conn_po);
    alt new_conn_result {
      ok(client_stream_ptr) {
        conn_port_new_tcp_socket(client_stream_ptr, iotask)
      }
      err(err_data) {
        result::err(err_data)
      }
    }
}

#[doc="
Identical to `net::tcp::conn_recv`, but ran on a new task

The recv'd tcp_socket is created with a new task on the current scheduler,
and given as a parameter to the provided callback

# Arguments

* `server_port` -- a `net::tcp::tcp_conn_port` that you wish to listen
on for an incoming connection
* `cb` -- a callback that will be ran, in a new task on the current scheduler,
once a new connection is recv'd. Its parameter:
  * A `result` object containing a `net::tcp::tcp_socket`, ready for immediate
    use, as the `ok` varient, or a `net::tcp::tcp_err_data` for the `err`
    variant
"]
fn conn_recv_spawn(server_port: tcp_conn_port,
                   +cb: fn~(result::result<tcp_socket, tcp_err_data>)) {
    let new_conn_po = (**server_port).new_conn_po;
    let iotask = (**server_port).iotask;
    let new_conn_result = comm::recv(new_conn_po);
    task::spawn {||
        let sock_create_result = alt new_conn_result {
          ok(client_stream_ptr) {
            conn_port_new_tcp_socket(client_stream_ptr, iotask)
          }
          err(err_data) {
            result::err(err_data)
          }
        };
        cb(sock_create_result);
    };
}

#[doc="
Check if a `net::tcp::tcp_conn_port` has one-or-more pending, new connections

This function behaves similarly to `comm::peek()`

# Arguments

* `server_port` -- a `net::tcp::tcp_conn_port` representing a server
connection

# Returns

`true` if there are one-or-more pending connections, `false` if there are
none.
"]
fn conn_peek(server_port: tcp_conn_port) -> bool {
    let new_conn_po = (**server_port).new_conn_po;
    comm::peek(new_conn_po)
}

#[doc="
Bind an incoming client connection to a `net::tcp::tcp_socket`

# Notes

It is safe to call `net::tcp::accept` _only_ within the context of the
`new_connect_cb` callback provided as the final argument to the
`net::tcp::listen` function.

The `new_conn` opaque value is provided _only_ as the first argument to the
`new_connect_cb` provided as a part of `net::tcp::listen`.
It can be safely sent to another task but it _must_ be
used (via `net::tcp::accept`) before the `new_connect_cb` call it was
provided to returns.

This implies that a port/chan pair must be used to make sure that the
`new_connect_cb` call blocks until an attempt to create a
`net::tcp::tcp_socket` is completed.

# Example

Here, the `new_conn` is used in conjunction with `accept` from within
a task spawned by the `new_connect_cb` passed into `listen`

~~~~~~~~~~~
net::tcp::listen(remote_ip, remote_port, backlog) {|new_conn, kill_ch|
    let cont_po = comm::port::<option<tcp_err_data>>();
    let cont_ch = comm::chan(cont_po);
    task::spawn {||
        let accept_result = net::tcp::accept(new_conn);
        if accept_result.is_failure() {
            comm::send(cont_ch, result::get_err(accept_result));
            // fail?
        }
        else {
            let sock = result::get(accept_result);
            comm::send(cont_ch, true);
            // do work here
        }
    };
    alt comm::recv(cont_po) {
      // shut down listen()
      some(err_data) { comm::send(kill_chan, some(err_data)) }
      // wait for next connection
      none {}
    }
};
~~~~~~~~~~~

# Arguments

* `new_conn` - an opaque value used to create a new `tcp_socket`

# Returns

* Success
  * On success, this function will return a `net::tcp::tcp_socket` as the
  `ok` variant of a `result`. The `net::tcp::tcp_socket` is anchored within
  the task that `accept` was called within for its lifetime.
* Failure
  * On failure, this function will return a `net::tcp::tcp_err_data` record
  as the `err` variant of a `result`.
"]
fn accept(new_conn: tcp_new_connection)
    -> result::result<tcp_socket, tcp_err_data> unsafe {

    alt new_conn{
      new_tcp_conn(server_handle_ptr) {
        let server_data_ptr = uv::ll::get_data_for_uv_handle(
            server_handle_ptr) as *tcp_listen_fc_data;
        let reader_po = comm::port::<result::result<[u8], tcp_err_data>>();
        let iotask = (*server_data_ptr).iotask;
        let stream_handle_ptr = malloc_uv_tcp_t();
        *(stream_handle_ptr as *mut uv::ll::uv_tcp_t) = uv::ll::tcp_t();
        let client_socket_data = @{
            reader_po: reader_po,
            reader_ch: comm::chan(reader_po),
            stream_handle_ptr : stream_handle_ptr,
            connect_req : uv::ll::connect_t(),
            write_req : uv::ll::write_t(),
            iotask : iotask
        };
        let client_socket_data_ptr = ptr::addr_of(*client_socket_data);
        let client_stream_handle_ptr =
            (*client_socket_data_ptr).stream_handle_ptr;

        let result_po = comm::port::<option<tcp_err_data>>();
        let result_ch = comm::chan(result_po);

        // UNSAFE LIBUV INTERACTION BEGIN
        // .. normally this happens within the context of
        // a call to uv::hl::interact.. but we're breaking
        // the rules here because this always has to be
        // called within the context of a listen() new_connect_cb
        // callback (or it will likely fail and drown your cat)
        log(debug, "in interact cb for tcp::accept");
        let loop_ptr = uv::ll::get_loop_for_uv_handle(
            server_handle_ptr);
        alt uv::ll::tcp_init(loop_ptr, client_stream_handle_ptr) {
          0i32 {
            log(debug, "uv_tcp_init successful for client stream");
            alt uv::ll::accept(
                server_handle_ptr as *libc::c_void,
                client_stream_handle_ptr as *libc::c_void) {
              0i32 {
                log(debug, "successfully accepted client connection");
                uv::ll::set_data_for_uv_handle(client_stream_handle_ptr,
                                               client_socket_data_ptr
                                                   as *libc::c_void);
                comm::send(result_ch, none);
              }
              _ {
                log(debug, "failed to accept client conn");
                comm::send(result_ch, some(
                    uv::ll::get_last_err_data(loop_ptr).to_tcp_err()));
              }
            }
          }
          _ {
            log(debug, "failed to init client stream");
            comm::send(result_ch, some(
                uv::ll::get_last_err_data(loop_ptr).to_tcp_err()));
          }
        }
        // UNSAFE LIBUV INTERACTION END
        alt comm::recv(result_po) {
          some(err_data) {
            result::err(err_data)
          }
          none {
            result::ok(tcp_socket(client_socket_data))
          }
        }
      }
    }
}

#[doc="
Bind to a given IP/port and listen for new connections

# Arguments

* `host_ip` - a `net::ip::ip_addr` representing a unique IP
(versions 4 or 6)
* `port` - a uint representing the port to listen on
* `backlog` - a uint representing the number of incoming connections
to cache in memory
* `hl_loop` - a `uv::hl::high_level_loop` that the tcp request will run on
* `on_establish_cb` - a callback that is evaluated if/when the listener
is successfully established. it takes no parameters
* `new_connect_cb` - a callback to be evaluated, on the libuv thread,
whenever a client attempts to conect on the provided ip/port. the
callback's arguments are:
    * `new_conn` - an opaque type that can be passed to
    `net::tcp::accept` in order to be converted to a `tcp_socket`.
    * `kill_ch` - channel of type `comm::chan<option<tcp_err_data>>`. this
    channel can be used to send a message to cause `listen` to begin
    closing the underlying libuv data structures.

# returns

a `result` instance containing empty data of type `()` on a
successful/normal shutdown, and a `tcp_err_data` record in the event
of listen exiting because of an error
"]
fn listen_for_conn(host_ip: ip::ip_addr, port: uint, backlog: uint,
          iotask: iotask,
          on_establish_cb: fn~(comm::chan<option<tcp_err_data>>),
          +new_connect_cb: fn~(tcp_new_connection,
                               comm::chan<option<tcp_err_data>>))
    -> result::result<(), tcp_err_data> unsafe {
    let stream_closed_po = comm::port::<()>();
    let kill_po = comm::port::<option<tcp_err_data>>();
    let kill_ch = comm::chan(kill_po);
    let server_stream = uv::ll::tcp_t();
    let server_stream_ptr = ptr::addr_of(server_stream);
    let server_data = {
        server_stream_ptr: server_stream_ptr,
        stream_closed_ch: comm::chan(stream_closed_po),
        kill_ch: kill_ch,
        new_connect_cb: new_connect_cb,
        iotask: iotask,
        mut active: true
    };
    let server_data_ptr = ptr::addr_of(server_data);

    let setup_po = comm::port::<option<tcp_err_data>>();
    let setup_ch = comm::chan(setup_po);
    iotask::interact(iotask) {|loop_ptr|
        let tcp_addr = ipv4_ip_addr_to_sockaddr_in(host_ip,
                                                   port);
        alt uv::ll::tcp_init(loop_ptr, server_stream_ptr) {
          0i32 {
            alt uv::ll::tcp_bind(server_stream_ptr,
                                 ptr::addr_of(tcp_addr)) {
              0i32 {
                alt uv::ll::listen(server_stream_ptr,
                                   backlog as libc::c_int,
                                   tcp_lfc_on_connection_cb) {
                  0i32 {
                    uv::ll::set_data_for_uv_handle(
                        server_stream_ptr,
                        server_data_ptr);
                    comm::send(setup_ch, none);
                  }
                  _ {
                    log(debug, "failure to uv_listen()");
                    let err_data = uv::ll::get_last_err_data(loop_ptr);
                    comm::send(setup_ch, some(err_data));
                  }
                }
              }
              _ {
                log(debug, "failure to uv_tcp_bind");
                let err_data = uv::ll::get_last_err_data(loop_ptr);
                comm::send(setup_ch, some(err_data));
              }
            }
          }
          _ {
            log(debug, "failure to uv_tcp_init");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            comm::send(setup_ch, some(err_data));
          }
        }
    };
    alt comm::recv(setup_po) {
      some(err_data) {
        // we failed to bind/list w/ libuv
        result::err(err_data.to_tcp_err())
      }
      none {
        on_establish_cb(kill_ch);
        let kill_result = comm::recv(kill_po);
        iotask::interact(iotask) {|loop_ptr|
            log(debug, #fmt("tcp::listen post-kill recv hl interact %?",
                            loop_ptr));
            (*server_data_ptr).active = false;
            uv::ll::close(server_stream_ptr, tcp_lfc_close_cb);
        };
        comm::recv(stream_closed_po);
        alt kill_result {
          // some failure post bind/listen
          some(err_data) {
            result::err(err_data)
          }
          // clean exit
          none {
            result::ok(())
          }
        }
      }
    }
}

#[doc="
Convenience methods extending `net::tcp::tcp_conn_port`
"]
impl conn_port_methods for tcp_conn_port {
    fn recv() -> result::result<tcp_socket, tcp_err_data> { conn_recv(self) }
    fn recv_spawn(+cb: fn~(result::result<tcp_socket,tcp_err_data>))
                  { conn_recv_spawn(self, cb); }
    fn peek() -> bool { conn_peek(self) }
}

#[doc="
Convenience methods extending `net::tcp::tcp_socket`
"]
impl sock_methods for tcp_socket {
    fn read_start() -> result::result<comm::port<
        result::result<[u8], tcp_err_data>>, tcp_err_data> {
        read_start(self)
    }
    fn read_stop() ->
        result::result<(), tcp_err_data> {
        read_stop(self)
    }
    fn read(timeout_msecs: uint) ->
        result::result<[u8], tcp_err_data> {
        read(self, timeout_msecs)
    }
    fn read_future(timeout_msecs: uint) ->
        future::future<result::result<[u8], tcp_err_data>> {
        read_future(self, timeout_msecs)
    }
    fn write(raw_write_data: [u8])
        -> result::result<(), tcp_err_data> {
        write(self, raw_write_data)
    }
    fn write_future(raw_write_data: [u8])
        -> future::future<result::result<(), tcp_err_data>> {
        write_future(self, raw_write_data)
    }
}
// INTERNAL API

// shared implementation for tcp::read
fn read_common_impl(socket_data: *tcp_socket_data, timeout_msecs: uint)
    -> result::result<[u8],tcp_err_data> unsafe {
    log(debug, "starting tcp::read");
    let iotask = (*socket_data).iotask;
    let rs_result = read_start_common_impl(socket_data);
    if result::is_failure(rs_result) {
        let err_data = result::get_err(rs_result);
        result::err(err_data)
    }
    else {
        log(debug, "tcp::read before recv_timeout");
        let read_result = if timeout_msecs > 0u {
            timer::recv_timeout(
               iotask, timeout_msecs, result::get(rs_result))
        } else {
            some(comm::recv(result::get(rs_result)))
        };
        log(debug, "tcp::read after recv_timeout");
        alt read_result {
          none {
            log(debug, "tcp::read: timed out..");
            let err_data = {
                err_name: "TIMEOUT",
                err_msg: "req timed out"
            };
            read_stop_common_impl(socket_data);
            result::err(err_data)
          }
          some(data_result) {
            log(debug, "tcp::read got data");
            read_stop_common_impl(socket_data);
            data_result
          }
        }
    }
}

// shared impl for read_stop
fn read_stop_common_impl(socket_data: *tcp_socket_data) ->
    result::result<(), tcp_err_data> unsafe {
    let stream_handle_ptr = (*socket_data).stream_handle_ptr;
    let stop_po = comm::port::<option<tcp_err_data>>();
    let stop_ch = comm::chan(stop_po);
    iotask::interact((*socket_data).iotask) {|loop_ptr|
        log(debug, "in interact cb for tcp::read_stop");
        alt uv::ll::read_stop(stream_handle_ptr as *uv::ll::uv_stream_t) {
          0i32 {
            log(debug, "successfully called uv_read_stop");
            comm::send(stop_ch, none);
          }
          _ {
            log(debug, "failure in calling uv_read_stop");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            comm::send(stop_ch, some(err_data.to_tcp_err()));
          }
        }
    };
    alt comm::recv(stop_po) {
      some(err_data) {
        result::err(err_data.to_tcp_err())
      }
      none {
        result::ok(())
      }
    }
}

// shared impl for read_start
fn read_start_common_impl(socket_data: *tcp_socket_data)
    -> result::result<comm::port<
        result::result<[u8], tcp_err_data>>, tcp_err_data> unsafe {
    let stream_handle_ptr = (*socket_data).stream_handle_ptr;
    let start_po = comm::port::<option<uv::ll::uv_err_data>>();
    let start_ch = comm::chan(start_po);
    log(debug, "in tcp::read_start before interact loop");
    iotask::interact((*socket_data).iotask) {|loop_ptr|
        log(debug, #fmt("in tcp::read_start interact cb %?", loop_ptr));
        alt uv::ll::read_start(stream_handle_ptr as *uv::ll::uv_stream_t,
                               on_alloc_cb,
                               on_tcp_read_cb) {
          0i32 {
            log(debug, "success doing uv_read_start");
            comm::send(start_ch, none);
          }
          _ {
            log(debug, "error attempting uv_read_start");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            comm::send(start_ch, some(err_data));
          }
        }
    };
    alt comm::recv(start_po) {
      some(err_data) {
        result::err(err_data.to_tcp_err())
      }
      none {
        result::ok((*socket_data).reader_po)
      }
    }
}

// shared implementation used by write and write_future
fn write_common_impl(socket_data_ptr: *tcp_socket_data,
                     raw_write_data: [u8])
    -> result::result<(), tcp_err_data> unsafe {
    let write_req_ptr = ptr::addr_of((*socket_data_ptr).write_req);
    let stream_handle_ptr =
        (*socket_data_ptr).stream_handle_ptr;
    let write_buf_vec =  [ uv::ll::buf_init(
        vec::unsafe::to_ptr(raw_write_data),
        vec::len(raw_write_data)) ];
    let write_buf_vec_ptr = ptr::addr_of(write_buf_vec);
    let result_po = comm::port::<tcp_write_result>();
    let write_data = {
        result_ch: comm::chan(result_po)
    };
    let write_data_ptr = ptr::addr_of(write_data);
    iotask::interact((*socket_data_ptr).iotask) {|loop_ptr|
        log(debug, #fmt("in interact cb for tcp::write %?", loop_ptr));
        alt uv::ll::write(write_req_ptr,
                          stream_handle_ptr,
                          write_buf_vec_ptr,
                          tcp_write_complete_cb) {
          0i32 {
            log(debug, "uv_write() invoked successfully");
            uv::ll::set_data_for_req(write_req_ptr, write_data_ptr);
          }
          _ {
            log(debug, "error invoking uv_write()");
            let err_data = uv::ll::get_last_err_data(loop_ptr);
            comm::send((*write_data_ptr).result_ch,
                       tcp_write_error(err_data.to_tcp_err()));
          }
        }
    };
    // FIXME: Instead of passing unsafe pointers to local data, and waiting
    // here for the write to complete, we should transfer ownership of
    // everything to the I/O task and let it deal with the aftermath,
    // so we don't have to sit here blocking.
    alt comm::recv(result_po) {
      tcp_write_success { result::ok(()) }
      tcp_write_error(err_data) { result::err(err_data.to_tcp_err()) }
    }
}

// various recv_* can use a tcp_conn_port can re-use this..
fn conn_port_new_tcp_socket(
    stream_handle_ptr: *uv::ll::uv_tcp_t,
    iotask: iotask)
    -> result::result<tcp_socket,tcp_err_data> unsafe {
    // tcp_nl_on_connection_cb
    let reader_po = comm::port::<result::result<[u8], tcp_err_data>>();
    let client_socket_data = @{
        reader_po : reader_po,
        reader_ch : comm::chan(reader_po),
        stream_handle_ptr : stream_handle_ptr,
        connect_req : uv::ll::connect_t(),
        write_req : uv::ll::write_t(),
        iotask : iotask
    };
    let client_socket_data_ptr = ptr::addr_of(*client_socket_data);
    comm::listen {|cont_ch|
        iotask::interact(iotask) {|loop_ptr|
            log(debug, #fmt("in interact cb 4 conn_port_new_tcp.. loop %?",
                loop_ptr));
            uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                           client_socket_data_ptr);
            cont_ch.send(());
        };
        cont_ch.recv()
    };
    result::ok(tcp_socket(client_socket_data))
}

enum tcp_new_connection {
    new_tcp_conn(*uv::ll::uv_tcp_t)
}

type tcp_conn_port_data = {
    server_stream: uv::ll::uv_tcp_t,
    stream_closed_po: comm::port<()>,
    stream_closed_ch: comm::chan<()>,
    iotask: iotask,
    new_conn_po: comm::port<result::result<*uv::ll::uv_tcp_t,
                                            tcp_err_data>>,
    new_conn_ch: comm::chan<result::result<*uv::ll::uv_tcp_t,
                                           tcp_err_data>>
};

type tcp_listen_fc_data = {
    server_stream_ptr: *uv::ll::uv_tcp_t,
    stream_closed_ch: comm::chan<()>,
    kill_ch: comm::chan<option<tcp_err_data>>,
    new_connect_cb: fn~(tcp_new_connection,
                        comm::chan<option<tcp_err_data>>),
    iotask: iotask,
    mut active: bool
};

crust fn tcp_lfc_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let server_data_ptr = uv::ll::get_data_for_uv_handle(
        handle) as *tcp_listen_fc_data;
    comm::send((*server_data_ptr).stream_closed_ch, ());
}

crust fn tcp_lfc_on_connection_cb(handle: *uv::ll::uv_tcp_t,
                                     status: libc::c_int) unsafe {
    let server_data_ptr = uv::ll::get_data_for_uv_handle(handle)
        as *tcp_listen_fc_data;
    let kill_ch = (*server_data_ptr).kill_ch;
    if (*server_data_ptr).active {
        alt status {
          0i32 {
            let new_conn = new_tcp_conn(handle);
            (*server_data_ptr).new_connect_cb(new_conn, kill_ch);
          }
          _ {
            let loop_ptr = uv::ll::get_loop_for_uv_handle(handle);
            comm::send(kill_ch,
                       some(uv::ll::get_last_err_data(loop_ptr)
                            .to_tcp_err()));
            (*server_data_ptr).active = false;
          }
        }
    }
}

crust fn tcp_nl_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let conn_data_ptr = uv::ll::get_data_for_uv_handle(
        handle) as *tcp_conn_port_data;
    comm::send((*conn_data_ptr).stream_closed_ch, ());
}

fn malloc_uv_tcp_t() -> *uv::ll::uv_tcp_t unsafe {
    rustrt::rust_uv_current_kernel_malloc(
        rustrt::rust_uv_helper_uv_tcp_t_size()) as *uv::ll::uv_tcp_t
}

crust fn tcp_nl_on_connection_cb(server_handle_ptr: *uv::ll::uv_tcp_t,
                                     status: libc::c_int) unsafe {
    let server_data_ptr = uv::ll::get_data_for_uv_handle(server_handle_ptr)
        as *tcp_conn_port_data;
    let new_conn_ch = (*server_data_ptr).new_conn_ch;
    let loop_ptr = uv::ll::get_loop_for_uv_handle(server_handle_ptr);
    alt status {
      0i32 {
        let client_stream_handle_ptr = malloc_uv_tcp_t();
        *(client_stream_handle_ptr as *mut uv::ll::uv_tcp_t) =
            uv::ll::tcp_t();
        alt uv::ll::tcp_init(loop_ptr, client_stream_handle_ptr) {
          0i32 {
            log(debug, "uv_tcp_init successful for client stream");
            alt uv::ll::accept(
                server_handle_ptr as *libc::c_void,
                client_stream_handle_ptr as *libc::c_void) {
              0i32 {
                log(debug, "successfully accepted client connection");
                comm::send(new_conn_ch,
                           result::ok(client_stream_handle_ptr));
              }
              _ {
                log(debug, "failed to accept client conn");
                comm::send(
                    new_conn_ch,
                    result::err(uv::ll::get_last_err_data(loop_ptr)
                        .to_tcp_err()));
              }
            }
          }
          _ {
            log(debug, "failed to init client stream");
            comm::send(
                new_conn_ch,
                result::err(uv::ll::get_last_err_data(loop_ptr)
                    .to_tcp_err()));
          }
        }
      }
      _ {
        comm::send(
            new_conn_ch,
            result::err(uv::ll::get_last_err_data(loop_ptr)
                .to_tcp_err()));
      }
    }
}

enum tcp_connect_result {
    tcp_connected(tcp_socket),
    tcp_connect_error(tcp_err_data)
}

enum tcp_write_result {
    tcp_write_success,
    tcp_write_error(tcp_err_data)
}

enum tcp_read_start_result {
    tcp_read_start_success(comm::port<tcp_read_result>),
    tcp_read_start_error(tcp_err_data)
}

enum tcp_read_result {
    tcp_read_data([u8]),
    tcp_read_done,
    tcp_read_err(tcp_err_data)
}

iface to_tcp_err_iface {
    fn to_tcp_err() -> tcp_err_data;
}

impl of to_tcp_err_iface for uv::ll::uv_err_data {
    fn to_tcp_err() -> tcp_err_data {
        { err_name: self.err_name, err_msg: self.err_msg }
    }
}

crust fn on_tcp_read_cb(stream: *uv::ll::uv_stream_t,
                    nread: libc::ssize_t,
                    ++buf: uv::ll::uv_buf_t) unsafe {
    log(debug, #fmt("entering on_tcp_read_cb stream: %? nread: %?",
                    stream, nread));
    let loop_ptr = uv::ll::get_loop_for_uv_handle(stream);
    let socket_data_ptr = uv::ll::get_data_for_uv_handle(stream)
        as *tcp_socket_data;
    alt nread as int {
      // incoming err.. probably eof
      -1 {
        let err_data = uv::ll::get_last_err_data(loop_ptr).to_tcp_err();
        log(debug, #fmt("on_tcp_read_cb: incoming err.. name %? msg %?",
                        err_data.err_name, err_data.err_msg));
        let reader_ch = (*socket_data_ptr).reader_ch;
        comm::send(reader_ch, result::err(err_data));
      }
      // do nothing .. unneeded buf
      0 {}
      // have data
      _ {
        // we have data
        log(debug, #fmt("tcp on_read_cb nread: %d", nread as int));
        let reader_ch = (*socket_data_ptr).reader_ch;
        let buf_base = uv::ll::get_base_from_buf(buf);
        let buf_len = uv::ll::get_len_from_buf(buf);
        let new_bytes = vec::unsafe::from_buf(buf_base, buf_len as uint);
        comm::send(reader_ch, result::ok(new_bytes));
      }
    }
    uv::ll::free_base_of_buf(buf);
    log(debug, "exiting on_tcp_read_cb");
}

crust fn on_alloc_cb(handle: *libc::c_void,
                     ++suggested_size: libc::size_t)
    -> uv::ll::uv_buf_t unsafe {
    log(debug, "tcp read on_alloc_cb!");
    let char_ptr = uv::ll::malloc_buf_base_of(suggested_size);
    log(debug, #fmt("tcp read on_alloc_cb h: %? char_ptr: %u sugsize: %u",
                     handle,
                     char_ptr as uint,
                     suggested_size as uint));
    uv::ll::buf_init(char_ptr, suggested_size as uint)
}

type tcp_socket_close_data = {
    closed_ch: comm::chan<()>
};

crust fn tcp_socket_dtor_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let data = uv::ll::get_data_for_uv_handle(handle)
        as *tcp_socket_close_data;
    let closed_ch = (*data).closed_ch;
    comm::send(closed_ch, ());
    log(debug, "tcp_socket_dtor_close_cb exiting..");
}

crust fn tcp_write_complete_cb(write_req: *uv::ll::uv_write_t,
                              status: libc::c_int) unsafe {
    let write_data_ptr = uv::ll::get_data_for_req(write_req)
        as *write_req_data;
    // FIXME: if instead of alt
    alt status {
      0i32 {
        log(debug, "successful write complete");
        comm::send((*write_data_ptr).result_ch, tcp_write_success);
      }
      _ {
        let stream_handle_ptr = uv::ll::get_stream_handle_from_write_req(
            write_req);
        let loop_ptr = uv::ll::get_loop_for_uv_handle(stream_handle_ptr);
        let err_data = uv::ll::get_last_err_data(loop_ptr);
        log(debug, "failure to write");
        comm::send((*write_data_ptr).result_ch, tcp_write_error(err_data));
      }
    }
}

type write_req_data = {
    result_ch: comm::chan<tcp_write_result>
};

type connect_req_data = {
    result_ch: comm::chan<conn_attempt>,
    closed_signal_ch: comm::chan<()>
};

crust fn stream_error_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let data = uv::ll::get_data_for_uv_handle(handle) as
        *connect_req_data;
    comm::send((*data).closed_signal_ch, ());
    log(debug, #fmt("exiting steam_error_close_cb for %?", handle));
}

crust fn tcp_connect_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    log(debug, #fmt("closed client tcp handle %?", handle));
}

crust fn tcp_connect_on_connect_cb(connect_req_ptr: *uv::ll::uv_connect_t,
                                   status: libc::c_int) unsafe {
    let conn_data_ptr = (uv::ll::get_data_for_req(connect_req_ptr)
                      as *connect_req_data);
    let result_ch = (*conn_data_ptr).result_ch;
    log(debug, #fmt("tcp_connect result_ch %?", result_ch));
    let tcp_stream_ptr =
        uv::ll::get_stream_handle_from_connect_req(connect_req_ptr);
    alt status {
      0i32 {
        log(debug, "successful tcp connection!");
        comm::send(result_ch, conn_success);
      }
      _ {
        log(debug, "error in tcp_connect_on_connect_cb");
        let loop_ptr = uv::ll::get_loop_for_uv_handle(tcp_stream_ptr);
        let err_data = uv::ll::get_last_err_data(loop_ptr);
        log(debug, #fmt("err_data %? %?", err_data.err_name,
                        err_data.err_msg));
        comm::send(result_ch, conn_failure(err_data));
        uv::ll::set_data_for_uv_handle(tcp_stream_ptr,
                                       conn_data_ptr);
        uv::ll::close(tcp_stream_ptr, stream_error_close_cb);
      }
    }
    log(debug, "leaving tcp_connect_on_connect_cb");
}

enum conn_attempt {
    conn_success,
    conn_failure(uv::ll::uv_err_data)
}

type tcp_socket_data = {
    reader_po: comm::port<result::result<[u8], tcp_err_data>>,
    reader_ch: comm::chan<result::result<[u8], tcp_err_data>>,
    stream_handle_ptr: *uv::ll::uv_tcp_t,
    connect_req: uv::ll::uv_connect_t,
    write_req: uv::ll::uv_write_t,
    iotask: iotask
};

// convert rust ip_addr to libuv's native representation
fn ipv4_ip_addr_to_sockaddr_in(input_ip: ip::ip_addr,
                               port: uint) -> uv::ll::sockaddr_in unsafe {
    // FIXME ipv6
    alt input_ip {
      ip::ipv4(_,_,_,_) {
        uv::ll::ip4_addr(ip::format_addr(input_ip), port as int)
      }
      ip::ipv6(_,_,_,_,_,_,_,_) {
        fail "FIXME ipv6 not yet supported";
      }
    }
}

#[cfg(test)]
mod test {
    // FIXME don't run on fbsd or linux 32 bit(#2064)
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
            fn test_gl_tcp_server_listener_and_client_ipv4() unsafe {
                impl_gl_tcp_ipv4_server_listener_and_client();
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
            fn test_gl_tcp_server_listener_and_client_ipv4() unsafe {
                impl_gl_tcp_ipv4_server_listener_and_client();
            }
        }
    }
    fn impl_gl_tcp_ipv4_server_and_client() {
        let hl_loop = uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 8888u;
        let expected_req = "ping";
        let expected_resp = "pong";

        let server_result_po = comm::port::<str>();
        let server_result_ch = comm::chan(server_result_po);

        let cont_po = comm::port::<()>();
        let cont_ch = comm::chan(cont_po);
        // server
        task::spawn_sched(task::manual_threads(1u)) {||
            let actual_req = comm::listen {|server_ch|
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
        comm::recv(cont_po);
        // client
        log(debug, "server started, firing up client..");
        let actual_resp = comm::listen {|client_ch|
            run_tcp_test_client(
                server_ip,
                server_port,
                expected_req,
                client_ch,
                hl_loop)
        };
        let actual_req = comm::recv(server_result_po);
        log(debug, #fmt("REQ: expected: '%s' actual: '%s'",
                       expected_req, actual_req));
        log(debug, #fmt("RESP: expected: '%s' actual: '%s'",
                       expected_resp, actual_resp));
        assert str::contains(actual_req, expected_req);
        assert str::contains(actual_resp, expected_resp);
    }
    fn impl_gl_tcp_ipv4_server_listener_and_client() {
        let hl_loop = uv::global_loop::get();
        let server_ip = "127.0.0.1";
        let server_port = 8889u;
        let expected_req = "ping";
        let expected_resp = "pong";

        let server_result_po = comm::port::<str>();
        let server_result_ch = comm::chan(server_result_po);

        let cont_po = comm::port::<()>();
        let cont_ch = comm::chan(cont_po);
        // server
        task::spawn_sched(task::manual_threads(1u)) {||
            let actual_req = comm::listen {|server_ch|
                run_tcp_test_server_listener(
                    server_ip,
                    server_port,
                    expected_resp,
                    server_ch,
                    cont_ch,
                    hl_loop)
            };
            server_result_ch.send(actual_req);
        };
        comm::recv(cont_po);
        // client
        log(debug, "server started, firing up client..");
        let actual_resp = comm::listen {|client_ch|
            run_tcp_test_client(
                server_ip,
                server_port,
                expected_req,
                client_ch,
                hl_loop)
        };
        let actual_req = comm::recv(server_result_po);
        log(debug, #fmt("REQ: expected: '%s' actual: '%s'",
                       expected_req, actual_req));
        log(debug, #fmt("RESP: expected: '%s' actual: '%s'",
                       expected_resp, actual_resp));
        assert str::contains(actual_req, expected_req);
        assert str::contains(actual_resp, expected_resp);
    }

    fn run_tcp_test_server(server_ip: str, server_port: uint, resp: str,
                          server_ch: comm::chan<str>,
                          cont_ch: comm::chan<()>,
                          iotask: iotask) -> str {

        task::spawn_sched(task::manual_threads(1u)) {||
            let server_ip_addr = ip::v4::parse_addr(server_ip);
            let listen_result =
                listen_for_conn(server_ip_addr, server_port, 128u,
                iotask,
                // on_establish_cb -- called when listener is set up
                {|kill_ch|
                    log(debug, #fmt("establish_cb %?",
                        kill_ch));
                    comm::send(cont_ch, ());
                },
                // risky to run this on the loop, but some users
                // will want the POWER
                {|new_conn, kill_ch|
                log(debug, "SERVER: new connection!");
                comm::listen {|cont_ch|
                    task::spawn_sched(task::manual_threads(1u)) {||
                        log(debug, "SERVER: starting worker for new req");

                        let accept_result = accept(new_conn);
                        log(debug, "SERVER: after accept()");
                        if result::is_failure(accept_result) {
                            log(debug, "SERVER: error accept connection");
                            let err_data = result::get_err(accept_result);
                            comm::send(kill_ch, some(err_data));
                            log(debug,
                                "SERVER/WORKER: send on err cont ch");
                            cont_ch.send(());
                        }
                        else {
                            log(debug,
                                "SERVER/WORKER: send on cont ch");
                            cont_ch.send(());
                            let sock = result::unwrap(accept_result);
                            log(debug, "SERVER: successfully accepted"+
                                "connection!");
                            let received_req_bytes = sock.read(0u);
                            alt received_req_bytes {
                              result::ok(data) {
                                server_ch.send(
                                    str::from_bytes(data));
                                log(debug, "SERVER: before write");
                                tcp_write_single(sock, str::bytes(resp));
                                log(debug, "SERVER: after write.. die");
                                comm::send(kill_ch, none);
                              }
                              result::err(err_data) {
                                log(debug, #fmt("SERVER: error recvd: %s %s",
                                    err_data.err_name, err_data.err_msg));
                                comm::send(kill_ch, some(err_data));
                                server_ch.send("");
                              }
                            }
                            log(debug, "SERVER: worker spinning down");
                        }
                    }
                    log(debug, "SERVER: waiting to recv on cont_ch");
                    cont_ch.recv()
                };
                log(debug, "SERVER: recv'd on cont_ch..leaving listen cb");
            });
            // err check on listen_result
            if result::is_failure(listen_result) {
                let err_data = result::get_err(listen_result);
                log(debug, #fmt("SERVER: exited abnormally name %s msg %s",
                                err_data.err_name, err_data.err_msg));
            }
        };
        let ret_val = server_ch.recv();
        log(debug, #fmt("SERVER: exited and got ret val: '%s'", ret_val));
        ret_val
    }

    fn run_tcp_test_server_listener(server_ip: str,
                                    server_port: uint, resp: str,
                                    server_ch: comm::chan<str>,
                                    cont_ch: comm::chan<()>,
                                    iotask: iotask) -> str {

        task::spawn_sched(task::manual_threads(1u)) {||
            let server_ip_addr = ip::v4::parse_addr(server_ip);
            let new_listener_result =
                new_listener(server_ip_addr, server_port, 128u, iotask);
            if result::is_failure(new_listener_result) {
                let err_data = result::get_err(new_listener_result);
                log(debug, #fmt("SERVER: exited abnormally name %s msg %s",
                                err_data.err_name, err_data.err_msg));
                fail "couldn't set up new listener";
            }
            let server_port = result::unwrap(new_listener_result);
            cont_ch.send(());
            // receive a single new connection.. normally this'd be
            // in a loop {}, but we're just going to take a single
            // client.. get their req, write a resp and then exit
            let new_conn_result = server_port.recv();
            if result::is_failure(new_conn_result) {
                let err_data = result::get_err(new_conn_result);
                log(debug, #fmt("SERVER: exited abnormally name %s msg %s",
                                err_data.err_name, err_data.err_msg));
                fail "couldn't recv new conn";
            }
            let sock = result::unwrap(new_conn_result);
            log(debug, "SERVER: successfully accepted"+
                "connection!");
            let received_req_bytes =
                sock.read(0u);
            alt received_req_bytes {
              result::ok(data) {
                server_ch.send(
                    str::from_bytes(data));
                log(debug, "SERVER: before write");
                tcp_write_single(sock, str::bytes(resp));
                log(debug, "SERVER: after write.. die");
              }
              result::err(err_data) {
                server_ch.send("");
              }
            }
        };
        let ret_val = server_ch.recv();
        log(debug, #fmt("SERVER: exited and got ret val: '%s'", ret_val));
        ret_val
    }

    fn run_tcp_test_client(server_ip: str, server_port: uint, resp: str,
                          client_ch: comm::chan<str>,
                          iotask: iotask) -> str {

        let server_ip_addr = ip::v4::parse_addr(server_ip);

        log(debug, "CLIENT: starting..");
        let connect_result = connect(server_ip_addr, server_port, iotask);
        if result::is_failure(connect_result) {
            log(debug, "CLIENT: failed to connect");
            let err_data = result::get_err(connect_result);
            log(debug, #fmt("CLIENT: connect err name: %s msg: %s",
                            err_data.err_name, err_data.err_msg));
            ""
        }
        else {
            let sock = result::unwrap(connect_result);
            let resp_bytes = str::bytes(resp);
            tcp_write_single(sock, resp_bytes);
            let read_result = sock.read(0u);
            if read_result.is_failure() {
                log(debug, "CLIENT: failure to read");
                ""
            }
            else {
                client_ch.send(str::from_bytes(read_result.get()));
                let ret_val = client_ch.recv();
                log(debug, #fmt("CLIENT: after client_ch recv ret: '%s'",
                   ret_val));
                ret_val
            }
        }
    }

    fn tcp_write_single(sock: tcp_socket, val: [u8]) {
        let write_result_future = sock.write_future(val);
        let write_result = write_result_future.get();
        if result::is_failure(write_result) {
            log(debug, "tcp_write_single: write failed!");
            let err_data = result::get_err(write_result);
            log(debug, #fmt("tcp_write_single err name: %s msg: %s",
                err_data.err_name, err_data.err_msg));
            // meh. torn on what to do here.
            fail "tcp_write_single failed";
        }
    }
}
