#[doc="
High-level interface to libuv's TCP functionality
"];

import ip = net_ip;
import comm::*;
import result::*;
import str::*;

// data
export tcp_socket, tcp_err_data;
// operations on a tcp_socket
export write, read_start, read_stop;
// tcp server stuff
export listen, accept;
// tcp client stuff
export connect;
// misc util
export is_responding;

#[doc="
Encapsulates an open TCP/IP connection through libuv

`tcp_socket` non-sendable and handles automatically closing the underlying libuv data structures when it goes out of scope.
"]
resource tcp_socket(socket_data: @tcp_socket_data) unsafe {
    let closed_po = comm::port::<()>();
    let closed_ch = comm::chan(closed_po);
    let close_data = {
        closed_ch: closed_ch
    };
    let close_data_ptr = ptr::addr_of(close_data);
    let stream_handle_ptr = ptr::addr_of((*socket_data).stream_handle);
    uv::hl::interact((*socket_data).hl_loop) {|loop_ptr|
        log(debug, #fmt("interact dtor for tcp_socket stream %? loop %?",
            stream_handle_ptr, loop_ptr));
        uv::ll::set_data_for_uv_handle(stream_handle_ptr,
                                       close_data_ptr);
        uv::ll::close(stream_handle_ptr, tcp_socket_dtor_close_cb);
    };
    comm::recv(closed_po);
    log(debug, "exiting dtor for tcp_socket");
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

* ip - The IP address (versions 4 or 6) of the remote host
* port - the unsigned integer of the desired remote host port

# Returns

A `result` that, if the operation succeeds, contains a `tcp_socket` that
can be used to send and receive data to/from the remote host. In the event
of failure, a `tcp_err_data` will be returned
"]
fn connect(input_ip: ip::ip_addr, port: uint)
    -> result::result<tcp_socket, tcp_err_data> unsafe {
    let result_po = comm::port::<conn_attempt>();
    let closed_signal_po = comm::port::<()>();
    let conn_data = {
        result_ch: comm::chan(result_po),
        closed_signal_ch: comm::chan(closed_signal_po)
    };
    let conn_data_ptr = ptr::addr_of(conn_data);
    let hl_loop = uv::global_loop::get();
    let reader_po = comm::port::<result::result<[u8], tcp_err_data>>();
    let socket_data = @{
        reader_po: reader_po,
        reader_ch: comm::chan(reader_po),
        stream_handle : uv::ll::tcp_t(),
        connect_req : uv::ll::connect_t(),
        write_req : uv::ll::write_t(),
        hl_loop: hl_loop
    };
    log(debug, #fmt("tcp_connect result_ch %?", conn_data.result_ch));
    // get an unsafe representation of our stream_handle_ptr that
    // we can send into the interact cb to be handled in libuv..
    let socket_data_ptr: *tcp_socket_data =
        ptr::addr_of(*socket_data);
    log(debug, #fmt("stream_handl_ptr outside interact %?",
        ptr::addr_of((*socket_data_ptr).stream_handle)));
    uv::hl::interact(hl_loop) {|loop_ptr|
        log(debug, "in interact cb for tcp client connect..");
        let stream_handle_ptr =
            ptr::addr_of((*socket_data_ptr).stream_handle);
        log(debug, #fmt("stream_handl_ptr in interact %?",
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
                                               socket_data_ptr);
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
Write binary data to a tcp stream

# Arguments

* sock - a `tcp_socket` to write to
* raw_write_data - a vector of `[u8]` that will be written to the stream.
This value must remain valid for the duration of the `write` call

# Returns

A `result` object with a `()` value, in the event of success, or a
`tcp_err_data` value in the event of failure
"]
fn write(sock: tcp_socket, raw_write_data: [[u8]])
    -> result::result<(), tcp_err_data> unsafe {
    let socket_data_ptr = ptr::addr_of(**sock);
    let write_req_ptr = ptr::addr_of((*socket_data_ptr).write_req);
    let stream_handle_ptr =
        ptr::addr_of((*socket_data_ptr).stream_handle);
    let write_buf_vec = iter::map_to_vec(raw_write_data) {|raw_bytes|
        uv::ll::buf_init(vec::unsafe::to_ptr(raw_bytes),
                         vec::len(raw_bytes))
    };
    let write_buf_vec_ptr = ptr::addr_of(write_buf_vec);
    let result_po = comm::port::<tcp_write_result>();
    let write_data = {
        result_ch: comm::chan(result_po)
    };
    let write_data_ptr = ptr::addr_of(write_data);
    uv::hl::interact((*socket_data_ptr).hl_loop) {|loop_ptr|
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
    alt comm::recv(result_po) {
      tcp_write_success { result::ok(()) }
      tcp_write_error(err_data) { result::err(err_data.to_tcp_err()) }
    }
}

#[doc="
Begin reading binary data from an open TCP connection.

# Arguments

* sock -- a `tcp_socket` for the connection to read from

# Returns

* A `result` instance that will either contain a
`comm::port<tcp_read_result>` that the user can read (and optionally, loop
on) from until `read_stop` is called, or a `tcp_err_data` record
"]
fn read_start(sock: tcp_socket)
    -> result::result<comm::port<
        result::result<[u8], tcp_err_data>>, tcp_err_data> unsafe {
    let stream_handle_ptr = ptr::addr_of((**sock).stream_handle);
    let start_po = comm::port::<option<uv::ll::uv_err_data>>();
    let start_ch = comm::chan(start_po);
    log(debug, "in tcp::read_start before interact loop");
    uv::hl::interact((**sock).hl_loop) {|loop_ptr|
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
        result::ok((**sock).reader_po)
      }
    }
}

#[doc="
Stop reading from an open TCP connection.
"]
fn read_stop(sock: tcp_socket) ->
    result::result<(), tcp_err_data> unsafe {
    let stream_handle_ptr = ptr::addr_of((**sock).stream_handle);
    let stop_po = comm::port::<option<tcp_err_data>>();
    let stop_ch = comm::chan(stop_po);
    uv::hl::interact((**sock).hl_loop) {|loop_ptr|
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

#[doc="
Bind to a given IP/port and listen for new connections

# Arguments

* `host_ip` - a `net::ip::ip_addr` representing a unique IP
(versions 4 or 6)
* `port` - a uint representing the port to listen on
* `backlog` - a uint representing the number of incoming connections
to cache in memory
* `new_connect_cb` - a callback to be evaluated, on the libuv thread,
whenever a client attempts to conect on the provided ip/port. The
callback's arguments are:
    * `new_conn` - an opaque type that can be passed to
    `net::tcp::accept` in order to be converted to a `tcp_socket`.
    * `kill_ch` - channel of type `comm::chan<option<tcp_err_data>>`. This
    channel can be used to send a message to cause `listen` to begin
    closing the underlying libuv data structures.

# Returns

A `result` instance containing empty data of type `()` on a successful
or normal shutdown, and a `tcp_err_data` record in the event of listen
exiting because of an error
"]
fn listen(host_ip: ip::ip_addr, port: uint, backlog: uint,
          new_connect_cb: fn~(tcp_new_connection,
                              comm::chan<option<tcp_err_data>>))
    -> result::result<(), tcp_err_data> unsafe {
    let stream_closed_po = comm::port::<()>();
    let kill_po = comm::port::<option<tcp_err_data>>();
    let server_stream = uv::ll::tcp_t();
    let server_stream_ptr = ptr::addr_of(server_stream);
    let hl_loop = uv::global_loop::get();
    let server_data = {
        server_stream_ptr: server_stream_ptr,
        stream_closed_ch: comm::chan(stream_closed_po),
        kill_ch: comm::chan(kill_po),
        new_connect_cb: new_connect_cb,
        hl_loop: hl_loop,
        mut active: true
    };
    let server_data_ptr = ptr::addr_of(server_data);

    let setup_po = comm::port::<option<tcp_err_data>>();
    let setup_ch = comm::chan(setup_po);
    uv::hl::interact(hl_loop) {|loop_ptr|
        let tcp_addr = ipv4_ip_addr_to_sockaddr_in(host_ip,
                                                   port);
        alt uv::ll::tcp_init(loop_ptr, server_stream_ptr) {
          0i32 {
            alt uv::ll::tcp_bind(server_stream_ptr,
                                 ptr::addr_of(tcp_addr)) {
              0i32 {
                alt uv::ll::listen(server_stream_ptr,
                                   backlog as libc::c_int,
                                   tcp_listen_on_connection_cb) {
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
    let mut kill_result: option<tcp_err_data> = none;
    alt comm::recv(setup_po) {
      some(err_data) {
        // we failed to bind/list w/ libuv
        result::err(err_data.to_tcp_err())
      }
      none {
        kill_result = comm::recv(kill_po);
        uv::hl::interact(hl_loop) {|loop_ptr|
            log(debug, #fmt("tcp::listen post-kill recv hl interact %?",
                            loop_ptr));
            (*server_data_ptr).active = false;
            uv::ll::close(server_stream_ptr, tcp_listen_close_cb);
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
        alt accept_result.is_failure() {
          false { comm::send(cont_ch, result::get_err(accept_result)); }
          true {
            let sock = result::get(accept_result);
            comm::send(cont_ch, true);
            // do work here
          }
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
            server_handle_ptr) as *tcp_server_data;
        let reader_po = comm::port::<result::result<[u8], tcp_err_data>>();
        let hl_loop = (*server_data_ptr).hl_loop;
        let client_socket_data = @{
            reader_po: reader_po,
            reader_ch: comm::chan(reader_po),
            stream_handle : uv::ll::tcp_t(),
            connect_req : uv::ll::connect_t(),
            write_req : uv::ll::write_t(),
            hl_loop: hl_loop
        };
        let client_socket_data_ptr = ptr::addr_of(*client_socket_data);
        let client_stream_handle_ptr = ptr::addr_of(
            (*client_socket_data_ptr).stream_handle);

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
                                               client_socket_data_ptr);
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
Attempt to open a TCP/IP connection on a remote host

The connection will (attempt to) be successfully established and then
disconnect immediately. It is useful to determine, simply, if a remote
host is responding, and that is all.

# Arguments

* `remote_ip` - an IP address (versions 4 or 6) for the remote host
* `remote_port` - a uint representing the port on the remote host to
connect to
* `timeout_msecs` - a timeout period, in miliseconds, to wait before
aborting the connection attempt

# Returns

A `bool` indicating success or failure. If a connection was established
to the remote host in the alloted timeout, `true` is returned. If the
host refused the connection, timed out or had some other error condition,
`false` is returned.
"]
fn is_responding(remote_ip: ip::ip_addr, remote_port: uint,
                timeout_msecs: uint) -> bool {
    log(debug, "entering is_responding");
    let connected_po = comm::port::<bool>();
    let connected_ch = comm::chan(connected_po);
    task::spawn {||
        log(debug, "in is_responding nested task");
        let connect_result = connect(remote_ip, remote_port);
        let connect_succeeded = result::is_success(connect_result);
        log(debug, #fmt("leaving is_responding nested task .. result %?",
           connect_succeeded));
        comm::send(connected_ch, connect_succeeded);
    };
    log(debug, "exiting is_responding");
    alt timer::recv_timeout(timeout_msecs, connected_po) {
      some(connect_succeeded) {
        log(debug, #fmt("connect succedded? %?", connect_succeeded));
        connect_succeeded }
      none {
        log(debug, "is_responding timed out on waiting to connect");
        false }
    }
}

// INTERNAL API

enum tcp_new_connection {
    new_tcp_conn(*uv::ll::uv_tcp_t)
}

type tcp_server_data = {
    server_stream_ptr: *uv::ll::uv_tcp_t,
    stream_closed_ch: comm::chan<()>,
    kill_ch: comm::chan<option<tcp_err_data>>,
    new_connect_cb: fn~(tcp_new_connection,
                        comm::chan<option<tcp_err_data>>),
    hl_loop: uv::hl::high_level_loop,
    mut active: bool
};

crust fn tcp_listen_close_cb(handle: *uv::ll::uv_tcp_t) unsafe {
    let server_data_ptr = uv::ll::get_data_for_uv_handle(
        handle) as *tcp_server_data;
    comm::send((*server_data_ptr).stream_closed_ch, ());
}

crust fn tcp_listen_on_connection_cb(handle: *uv::ll::uv_tcp_t,
                                     status: libc::c_int) unsafe {
    let server_data_ptr = uv::ll::get_data_for_uv_handle(handle)
        as *tcp_server_data;
    let kill_ch = (*server_data_ptr).kill_ch;
    alt (*server_data_ptr).active {
      true {
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
      _ {
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
    alt nread {
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
        log(debug, #fmt("tcp on_read_cb nread: %d", nread));
        let reader_ch = (*socket_data_ptr).reader_ch;
        let buf_base = uv::ll::get_base_from_buf(buf);
        let buf_len = uv::ll::get_len_from_buf(buf);
        let new_bytes = vec::unsafe::from_buf(buf_base, buf_len);
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
    uv::ll::buf_init(char_ptr, suggested_size)
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
    stream_handle: uv::ll::uv_tcp_t,
    connect_req: uv::ll::uv_connect_t,
    write_req: uv::ll::uv_write_t,
    hl_loop: uv::hl::high_level_loop
};

// convert rust ip_addr to libuv's native representation
fn ipv4_ip_addr_to_sockaddr_in(input_ip: ip::ip_addr,
                               port: uint) -> uv::ll::sockaddr_in unsafe {
    // FIXME ipv6
    alt input_ip {
      ip::ipv4(_,_,_,_) {
        uv::ll::ip4_addr(ip::format_addr(input_ip), port as int)
      }
      _ {
        fail "FIXME ipv6 not yet supported";
      }
    }
}

//#[cfg(test)]
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
        }
        #[cfg(target_arch="x86")]
        mod impl32 {
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_gl_tcp_server_and_client_ipv4() unsafe {
                impl_gl_tcp_ipv4_server_and_client();
            }
        }
    }
    fn impl_gl_tcp_ipv4_server_and_client() {
        let server_ip = "127.0.0.1";
        let server_port = 8888u;
        let expected_req = "ping";
        let expected_resp = "pong";

        let server_result_po = comm::port::<str>();
        let server_result_ch = comm::chan(server_result_po);
        // server
        task::spawn_sched(task::manual_threads(1u)) {||
            let actual_req = comm::listen {|server_ch|
                run_tcp_test_server(
                    server_ip,
                    server_port,
                    expected_resp,
                    server_ch)
            };
            server_result_ch.send(actual_req);
        };
        // client
        log(debug, "server started, firing up client..");
        let actual_resp = comm::listen {|client_ch|
            log(debug, "before client sleep");
            timer::sleep(2u);
            log(debug, "after client sleep");
            run_tcp_test_client(
                server_ip,
                server_port,
                expected_req,
                client_ch)
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
                          server_ch: comm::chan<str>) -> str {

        task::spawn_sched(task::manual_threads(1u)) {||
            let server_ip_addr = ip::v4::parse_addr(server_ip);
            let listen_result = listen(server_ip_addr, server_port, 128u)
                // this callback is ran on the loop.
                // .. should it go?
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
                            let received_req_bytes =
                                tcp_read_single(sock);
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
            };
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
    
    fn run_tcp_test_client(server_ip: str, server_port: uint, resp: str,
                          client_ch: comm::chan<str>) -> str {

        let server_ip_addr = ip::v4::parse_addr(server_ip);

        log(debug, "CLIENT: starting..");
        let connect_result = connect(server_ip_addr, server_port);
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
            let read_result = tcp_read_single(sock);
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

    fn tcp_read_single(sock: tcp_socket)
        -> result::result<[u8],tcp_err_data> {
        log(debug, "starting tcp_read_single");
        let rs_result = read_start(sock);
        if result::is_failure(rs_result) {
            let err_data = result::get_err(rs_result);
            result::err(err_data)
        }
        else {
            log(debug, "before recv_timeout");
            let read_result = timer::recv_timeout(
                2000u, result::get(rs_result));
            log(debug, "after recv_timeout");
            alt read_result {
              none {
                log(debug, "tcp_read_single: timed out..");
                let err_data = {
                    err_name: "TIMEOUT",
                    err_msg: "req timed out"
                };
                result::err(err_data)
              }
              some(data_result) {
                log(debug, "tcp_read_single: got data");
                data_result
              }
            }
        }
    }

    fn tcp_write_single(sock: tcp_socket, val: [u8]) {
        let write_result = write(sock, [val]);
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
