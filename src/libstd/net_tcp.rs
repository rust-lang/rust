#[doc="
High-level interface to libuv's TCP functionality
"];

import ip = net_ip;

export tcp_connect_result, tcp_write_result;
export connect, write;

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

enum tcp_connect_result {
    tcp_connected(tcp_socket),
    tcp_connect_error(uv::ll::uv_err_data)
}

enum tcp_write_result {
    tcp_write_success,
    tcp_write_error(uv::ll::uv_err_data)
}

#[doc="
Initiate a client connection over TCP/IP

# Arguments

* ip - The IP address (versions 4 or 6) of the remote host
* port - the unsigned integer of the desired remote host port

# Returns

A `tcp_connect_result` that can be used to determine the connection and,
if successful, send and receive data to/from the remote host
"]
fn connect(input_ip: ip::ip_addr, port: uint) -> tcp_connect_result unsafe {
    let result_po = comm::port::<conn_attempt>();
    let closed_signal_po = comm::port::<()>();
    let conn_data = {
        result_ch: comm::chan(result_po),
        closed_signal_ch: comm::chan(closed_signal_po)
    };
    let conn_data_ptr = ptr::addr_of(conn_data);
    let hl_loop = uv::global_loop::get();
    let socket_data = @{
        reader_port: comm::port::<[u8]>(),
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
                               conn_failure(err_data));
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
                       conn_failure(err_data));
          }
        }
    };
    alt comm::recv(result_po) {
      conn_success {
        log(debug, "tcp::connect - received success on result_po");
        tcp_connected(tcp_socket(socket_data))
      }
      conn_failure(err_data) {
        comm::recv(closed_signal_po);
        log(debug, "tcp::connect - received failure on result_po");
        tcp_connect_error(err_data)
      }
    }
}

#[doc="
Write binary data to a tcp stream
"]
fn write(sock: tcp_socket, raw_write_data: [[u8]]) -> tcp_write_result
    unsafe {
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
                       tcp_write_error(err_data));
          }
        }
    };
    comm::recv(result_po)
}



// INTERNAL API

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
    reader_port: comm::port<[u8]>,
    stream_handle: uv::ll::uv_tcp_t,
    connect_req: uv::ll::uv_connect_t,
    write_req: uv::ll::uv_write_t,
    hl_loop: uv::hl::high_level_loop
};

// convert rust ip_addr to libuv's native representation
fn ipv4_ip_addr_to_sockaddr_in(input: ip::ip_addr,
                               port: uint) -> uv::ll::sockaddr_in unsafe {
    uv::ll::ip4_addr(ip::format_addr(input), port as int)
}

#[cfg(test)]
mod test {
    #[test]
    fn test_gl_tcp_ipv4_request() {
        let ip_str = "173.194.79.99";
        let port = 80u;
        let expected_read_msg = "foo";
        let actual_write_msg = "GET / HTTP/1.1\r\n\r\n";
        let host_ip = ip::v4::parse_addr(ip_str);

        let data_po = comm::port::<[u8]>();
        let data_ch = comm::chan(data_po);
        
        alt connect(host_ip, port) {
          tcp_connected(sock) {
            log(debug, "successful tcp connect");
            let mut write_data: [[u8]] = [];
            let write_data = [str::as_bytes(actual_write_msg) {|str_bytes|
                str_bytes
            }];
            alt write(sock, write_data) {
              tcp_write_success {
                log(debug, "tcp::write successful");
                /*
                let mut total_read_data: [u8] = [];
                let reader_po = read_start(sock);
                loop {
                    alt comm::recv(reader_po) {
                      new_read_data(data) {
                        total_read_data += data;
                        // theoretically, we could keep iterating, here, if
                        // we expect the server on the other end to keep
                        // streaming/chunking data to us, but..
                        read_stop(tcp_stream);
                        break;
                      }
                      done_reading {
                        break;
                      }
                      error {
                        fail "erroring occured during read attempt.."
                            + "FIXME need info";
                      }
                    }
                }
                comm::send(data_ch, total_read_data);
                */
              }
              tcp_write_error(err_data) {
                log(debug, "tcp_write_error received..");
                log(debug, #fmt("tcp write error: %? %?", err_data.err_name,
                               err_data.err_msg));
                assert false;
              }
            }
          }
          tcp_connect_error(err_data) {
            log(debug, "tcp_connect_error received..");
            log(debug, #fmt("tcp connect error: %? %?", err_data.err_name,
                           err_data.err_msg));
            assert false;
          }
        }

        let actual_data = comm::recv(data_po);
        let resp = str::from_bytes(actual_data);
        log(debug, "DATA RECEIVED: "+resp);
    }
}