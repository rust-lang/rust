#[doc="
High-level interface to libuv's TCP functionality
"];

import ip = net_ip;

export tcp_socket, tcp_err_data;
export connect, write, read_start, read_stop, listen, accept;

#[doc="
Encapsulates an open TCP/IP connection through libuv
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
whenever a client attempts to conect on the provided ip/port
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

It is safe to call `net::tcp::accept` _only_ within the callback
provided as the final argument of the `net::tcp::listen` function.

The `new_conn` opaque value provided _only_ as the first argument to the
`new_connect_cb`. It can be safely sent to another task but it _must_ be
used (via `net::tcp::accept`) before the `new_connect_cb` call it was
provided within returns.

This means that a port/chan pair must be used to make sure that the
`new_connect_cb` call blocks until an attempt to create a
`net::tcp::tcp_socket` is completed.

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
        let hl_loop = (*server_data_ptr).hl_loop;// FIXME
        let reader_po = comm::port::<result::result<[u8], tcp_err_data>>();
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
        uv::hl::interact(hl_loop) {|loop_ptr|
            log(debug, "in interact cb for tcp::accept");
            alt uv::ll::tcp_init(loop_ptr, client_stream_handle_ptr) {
              0i32 {
                log(debug, "uv_tcp_init successful for client stream");
                alt uv::ll::accept(server_handle_ptr,
                                   client_stream_handle_ptr) {
                  0i32 {
                    log(debug, "successfully accepted client connection");
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
        };
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
    log(debug, "entering on_tcp_read_cb");
    let loop_ptr = uv::ll::get_loop_for_uv_handle(stream);
    let socket_data_ptr = uv::ll::get_data_for_uv_handle(stream)
        as *tcp_socket_data;
    let reader_ch = (*socket_data_ptr).reader_ch;
    alt nread {
      // incoming err.. probably eof
      -1 {
        let err_data = uv::ll::get_last_err_data(loop_ptr);
        comm::send(reader_ch, result::err(err_data.to_tcp_err()));
      }
      // do nothing .. unneeded buf
      0 {}
      // have data
      _ {
        // we have data
        log(debug, #fmt("tcp on_read_cb nread: %d", nread));
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
    #[test]
    fn test_gl_tcp_ipv4_client() {
        let ip_str = "173.194.79.99";
        let port = 80u;
        let write_input = "GET / HTTP/1.1\r\n\r\n";
        let read_output =
            impl_gl_tcp_ipv4_client(ip_str, port, write_input);
        log(debug, "DATA RECEIVED: "+read_output);
    }

    #[test]
    fn test_gl_tcp_ipv4_server() {
        let server_ip = "127.0.0.1";
        let server_port = 8888u;
        let kill_str = "asdf";
        let resp_str = "hw";

        let result_po = comm::port::<str>();
        let result_ch = comm::chan(result_po);
        task::spawn_sched(task::manual_threads(4u)) {||
            let inner_result_po = comm::port::<str>();
            let inner_result_ch = comm::chan(inner_result_po);

            impl_gl_tcp_ipv4_server(server_ip, server_port,
                                    kill_str, resp_str,
                                    inner_result_ch);
            let result_str = comm::recv(inner_result_po);
            comm::send(result_ch, result_str);
        };
        let output = comm::recv(result_po);
        log(debug, #fmt("RECEIVED REQ %? FROM USER", output));
    }

    fn impl_gl_tcp_ipv4_server(host_str: str, port: uint,
                               kill_str: str, resp_str: str,
                              output_ch: comm::chan<str>) {
        let host_ip = ip::v4::parse_addr(host_str);
        log(debug, "about to enter listen() call for test server");
        listen(host_ip, port, 128u) {|new_conn, kill_ch|
            // this is a callback that is going to be invoked on the
            // loop's thread (can't be avoided).
            let cont_po = comm::port::<()>();
            let cont_ch = comm::chan(cont_po);
            task::spawn {||
                log(debug, "starting worker for incoming req");

                // work loop
                let accept_result = accept(new_conn);
                if result::is_failure(accept_result) {
                    // accept failed..
                    log(debug,"accept in worker task failed");
                    comm::send(kill_ch,
                               some(result::get_err(accept_result)
                                    .to_tcp_err()));
                }
                // accept() succeeded, let the task that is
                // listen()'ing know so it can continue and
                // unblock libuv..
                comm::send(cont_ch, ());

                // our precious sock.. from here on out, things
                // match the tcp request/client api, as they're
                // both on tcp_sockets at this point..
                let sock = result::unwrap(accept_result);
                let req_bytes = single_read_bytes_from(sock);
                let req_str = str::from_bytes(req_bytes);
                if str::contains(req_str, kill_str) {
                    // our signal to shut down the tcp
                    // server was received. shut it down.
                    comm::send(kill_ch, none);
                }
                write_single_str(sock, resp_str);

                comm::send(output_ch, req_str);
                // work's complete, let socket close..
                log(debug, "exiting worker");
            };

            comm::recv(cont_po);
        };
        log(debug, "exiting listen() block for test server");
    }

    fn impl_gl_tcp_ipv4_client(ip_str: str, port: uint,
                               write_input: str) -> str {
        // pre-connection/input data
        let host_ip = ip::v4::parse_addr(ip_str);

        // connect to remote host
        let connect_result = connect(host_ip, port);
        if result::is_failure(connect_result) {
            let err_data = result::get_err(connect_result);
            log(debug, "tcp_connect_error received..");
            log(debug, #fmt("tcp connect error: %? %?", err_data.err_name,
                           err_data.err_msg));
            assert false;
        }

        // this is our tcp_socket resource instance. It's dtor will
        // clean-up/close the underlying TCP stream when the fn scope
        // ends
        let sock = result::unwrap(connect_result);
        log(debug, "successful tcp connect");

        // set up write data
        let write_data = [str::as_bytes(write_input) {|str_bytes|
            str_bytes
        }];

        // write data to tcp socket
        let write_result = write(sock, write_data);
        if result::is_failure(write_result) {
            let err_data = result::get_err(write_result);
            log(debug, "tcp_write_error received..");
            log(debug, #fmt("tcp write error: %? %?", err_data.err_name,
                           err_data.err_msg));
            assert false;
        }
        log(debug, "tcp::write successful");

        // set up read data
        let mut total_read_data: [u8] = [];
        let read_start_result = read_start(sock);
        if result::is_failure(read_start_result) {
            let err_data = result::get_err(read_start_result);
            log(debug, "tcp read_start err received..");
            log(debug, #fmt("read_start error: %? %?", err_data.err_name,
                           err_data.err_msg));
            assert false;
        }
        let reader_po = result::get(read_start_result);
        loop {
            let read_data_result = comm::recv(reader_po);
            if result::is_failure(read_data_result) {
                let err_data = result::get_err(read_data_result);
                log(debug, "read error data recv'd");
                log(debug, #fmt("read error: %? %?",
                                err_data.err_name,
                                err_data.err_msg));
                assert false;
            }
            let new_data = result::unwrap(read_data_result);
            total_read_data += new_data;
            // theoretically, we could keep iterating, if
            // we expect the server on the other end to keep
            // streaming/chunking data to us, but..
            let read_stop_result = read_stop(sock);
            if result::is_failure(read_stop_result) {
                let err_data = result::get_err(read_stop_result);
                log(debug, "error while calling read_stop");
                log(debug, #fmt("read_stop error: %? %?",
                                err_data.err_name,
                                err_data.err_msg));
                assert false;
            }
            break;
        }
        str::from_bytes(total_read_data)
    }

    fn single_read_bytes_from(sock: tcp_socket) -> [u8] {
        let mut total_read_data: [u8] = [];
        let read_start_result = read_start(sock);
        if result::is_failure(read_start_result) {
            let err_data = result::get_err(read_start_result);
            log(debug, "srbf tcp read_start err received..");
            log(debug, #fmt("srbf read_start error: %? %?",
                            err_data.err_name,
                           err_data.err_msg));
            assert false;
        }
        let reader_po = result::get(read_start_result);

        let read_data_result = comm::recv(reader_po);
        if result::is_failure(read_data_result) {
            let err_data = result::get_err(read_data_result);
            log(debug, "srbf read error data recv'd");
            log(debug, #fmt("srbf read error: %? %?",
                            err_data.err_name,
                            err_data.err_msg));
            assert false;
        }
        let new_data = result::unwrap(read_data_result);
        total_read_data += new_data;
        // theoretically, we could keep iterating, if
        // we expect the server on the other end to keep
        // streaming/chunking data to us, but..
        let read_stop_result = read_stop(sock);
        if result::is_failure(read_stop_result) {
            let err_data = result::get_err(read_stop_result);
            log(debug, "srbf error while calling read_stop");
            log(debug, #fmt("srbf read_stop error: %? %?",
                            err_data.err_name,
                            err_data.err_msg));
            assert false;
        }
        total_read_data
    }

    fn write_single_str(sock: tcp_socket, write_input: str) {
        // set up write data
        let write_data = [str::as_bytes(write_input) {|str_bytes|
            str_bytes
        }];

        // write data to tcp socket
        let write_result = write(sock, write_data);
        if result::is_failure(write_result) {
            let err_data = result::get_err(write_result);
            log(debug, "wss tcp_write_error received..");
            log(debug, #fmt("wss tcp write error: %? %?",
                            err_data.err_name,
                            err_data.err_msg));
            assert false;
        }
        log(debug, "wss tcp::write successful");
    }
}
