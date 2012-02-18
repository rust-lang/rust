// Some temporary libuv hacks for servo

// UV2
enum uv_operation {
    op_hw()
}

enum uv_msg {
    // requests from library users
    msg_run(comm::chan<bool>),
    msg_run_in_bg(),
    msg_loop_delete(),
    msg_async_init([u8], fn~()),
    msg_async_send([u8]),
    msg_hw(),

    // dispatches from libuv
    uv_hw()
}

type uv_loop_data = {
    operation_port: comm::port<uv_operation>,
    rust_loop_chan: comm::chan<uv_msg>
};

type uv_loop = comm::chan<uv_msg>;

enum uv_handle {
    handle([u8], *ctypes::void)
}

#[nolink]
native mod rustrt {
    fn rust_uvtmp_create_thread() -> thread;
    fn rust_uvtmp_start_thread(thread: thread);
    fn rust_uvtmp_join_thread(thread: thread);
    fn rust_uvtmp_delete_thread(thread: thread);
    fn rust_uvtmp_connect(
        thread: thread,
        req_id: u32,
        ip: str::sbuf,
        chan: comm::chan<iomsg>) -> connect_data;
    fn rust_uvtmp_close_connection(thread: thread, req_id: u32);
    fn rust_uvtmp_write(
        thread: thread,
        req_id: u32,
        buf: *u8,
        len: ctypes::size_t,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_read_start(
        thread: thread,
        req_id: u32,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_timer(
        thread: thread,
        timeout: u32,
        req_id: u32,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_delete_buf(buf: *u8);
    fn rust_uvtmp_get_req_id(cd: connect_data) -> u32;

    fn rust_uvtmp_uv_loop_new() -> *ctypes::void;
    fn rust_uvtmp_uv_loop_set_data(
        loop: *ctypes::void,
        data: *uv_loop_data);
    fn rust_uvtmp_uv_bind_op_cb(loop: *ctypes::void, cb: *u8) -> *ctypes::void;
    fn rust_uvtmp_uv_run(loop_handle: *ctypes::void);
    fn rust_uvtmp_uv_async_send(handle: *ctypes::void);
}

mod uv {
    export loop_new, run, run_in_bg, hw;

    // public functions
    fn loop_new() -> uv_loop unsafe {
        let ret_recv_port: comm::port<uv_loop> =
            comm::port();
        let ret_recv_chan: comm::chan<uv_loop> =
            comm::chan(ret_recv_port);

        task::spawn_sched(3u) {||
            // our beloved uv_loop_t ptr
            let loop_handle = rustrt::
                rust_uvtmp_uv_loop_new();

            // this port/chan pair are used to send messages to
            // libuv. libuv processes any pending messages on the
            // port (via crust) after receiving an async "wakeup"
            // on a special uv_async_t handle created below
            let operation_port = comm::port::<uv_operation>();
            let operation_chan = comm::chan::<uv_operation>(
                operation_port);

            // this port/chan pair as used in the while() loop
            // below. It takes dispatches, originating from libuv
            // callbacks, to invoke handles registered by the
            // user
            let rust_loop_port = comm::port::<uv_msg>();
            let rust_loop_chan =
                comm::chan::<uv_msg>(rust_loop_port);
            // let the task-spawner return
            comm::send(ret_recv_chan, copy(rust_loop_chan));

            // create our "special" async handle that will
            // allow all operations against libuv to be
            // "buffered" in the operation_port, for processing
            // from the thread that libuv runs on
            let loop_data: uv_loop_data = {
                operation_port: operation_port,
                rust_loop_chan: rust_loop_chan
            };
            rustrt::rust_uvtmp_uv_loop_set_data(
                loop_handle,
                ptr::addr_of(loop_data)); // pass an opaque C-ptr
                                          // to libuv, this will be
                                          // in the process_operation
                                          // crust fn
            let async_handle = rustrt::rust_uvtmp_uv_bind_op_cb(
                loop_handle,
                process_operation);

            // all state goes here
            let handles: map::map<[u8], uv_handle> =
                map::new_bytes_hash();

            // the main loop that this task blocks on.
            // should have the same lifetime as the C libuv
            // event loop.
            let keep_going = true;
            while (keep_going) {
                alt comm::recv(rust_loop_port) {
                  msg_run(end_chan) {
                    // start the libuv event loop
                    // we'll also do a uv_async_send with
                    // the operation handle to have the
                    // loop process any pending operations
                    // once its up and running
                    task::spawn_sched(1u) {||
                        // this call blocks
                        rustrt::rust_uvtmp_uv_run(loop_handle);
                        // when we're done, msg the
                        // end chan
                        comm::send(end_chan, true);
                    };
                  }
                  msg_run_in_bg {
                    task::spawn_sched(1u) {||
                        // this call blocks
                        rustrt::rust_uvtmp_uv_run(loop_handle);
                    };
                  }
                  msg_hw() {
                    comm::send(operation_chan, op_hw);
                    io::println("CALLING ASYNC_SEND FOR HW");
                    rustrt::rust_uvtmp_uv_async_send(async_handle);
                  }
                  uv_hw() {
                    io::println("HELLO WORLD!!!");
                  }

                  ////// STUBS ///////
                  msg_loop_delete {
                    // delete the event loop's c ptr
                    // this will of course stop any
                    // further processing
                  }
                  msg_async_init(id, callback) {
                    // create a new async handle
                    // with the id as the handle's
                    // data and save the callback for
                    // invocation on msg_async_send
                  }
                  msg_async_send(id) {
                    // get the callback matching the
                    // supplied id and invoke it
                  }

                  _ { fail "unknown form of uv_msg received"; }
                }
            }
        };
        ret comm::recv(ret_recv_port);
    }

    fn run(loop: uv_loop) {
        let end_port = comm::port::<bool>();
        let end_chan = comm::chan::<bool>(end_port);
        comm::send(loop, msg_run(end_chan));
        comm::recv(end_port);
    }

    fn run_in_bg(loop: uv_loop) {
        comm::send(loop, msg_run_in_bg);
    }

    fn hw(loop: uv_loop) {
        comm::send(loop, msg_hw);
    }

    // internal functions

    // crust
    crust fn process_operation(data: *uv_loop_data) unsafe {
        io::println("IN PROCESS_OPERATION");
        let op_port = (*data).operation_port;
        let loop_chan = (*data).rust_loop_chan;
        let op_pending = comm::peek(op_port);
        while(op_pending) {
            io::println("OPERATION PENDING!");
            alt comm::recv(op_port) {
              op_hw() {
                io::println("GOT OP_HW IN CRUST");
                comm::send(loop_chan, uv_hw);
              }
              _ { fail "unknown form of uv_operation received"; }
            }
            op_pending = comm::peek(op_port);
        }
        io::println("NO MORE OPERATIONS PENDING!");
    }
}

#[test]
fn uvtmp_uv_test_hello_world() {
    let test_loop = uv::loop_new();
    uv::hw(test_loop);
    uv::run(test_loop);
}

// END OF UV2

type thread = *ctypes::void;

type connect_data = *ctypes::void;

enum iomsg {
    whatever,
    connected(connect_data),
    wrote(connect_data),
    read(connect_data, *u8, ctypes::ssize_t),
    timer(u32),
    exit
}

fn create_thread() -> thread {
    rustrt::rust_uvtmp_create_thread()
}

fn start_thread(thread: thread) {
    rustrt::rust_uvtmp_start_thread(thread)
}

fn join_thread(thread: thread) {
    rustrt::rust_uvtmp_join_thread(thread)
}

fn delete_thread(thread: thread) {
    rustrt::rust_uvtmp_delete_thread(thread)
}

fn connect(thread: thread, req_id: u32,
           ip: str, ch: comm::chan<iomsg>) -> connect_data {
    str::as_buf(ip) {|ipbuf|
        rustrt::rust_uvtmp_connect(thread, req_id, ipbuf, ch)
    }
}

fn close_connection(thread: thread, req_id: u32) {
    rustrt::rust_uvtmp_close_connection(thread, req_id);
}

fn write(thread: thread, req_id: u32, bytes: [u8],
         chan: comm::chan<iomsg>) unsafe {
    rustrt::rust_uvtmp_write(
        thread, req_id, vec::to_ptr(bytes), vec::len(bytes), chan);
}

fn read_start(thread: thread, req_id: u32,
              chan: comm::chan<iomsg>) {
    rustrt::rust_uvtmp_read_start(thread, req_id, chan);
}

fn timer_start(thread: thread, timeout: u32, req_id: u32,
              chan: comm::chan<iomsg>) {
    rustrt::rust_uvtmp_timer(thread, timeout, req_id, chan);
}

fn delete_buf(buf: *u8) {
    rustrt::rust_uvtmp_delete_buf(buf);
}

fn get_req_id(cd: connect_data) -> u32 {
    ret rustrt::rust_uvtmp_get_req_id(cd);
}

#[test]
fn test_start_stop() {
    let thread = create_thread();
    start_thread(thread);
    join_thread(thread);
    delete_thread(thread);
}

#[test]
#[ignore]
fn test_connect() {
    let thread = create_thread();
    start_thread(thread);
    let port = comm::port();
    let chan = comm::chan(port);
    connect(thread, 0u32, "74.125.224.146", chan);
    alt comm::recv(port) {
      connected(cd) {
        close_connection(thread, 0u32);
      }
      _ { fail "test_connect: port isn't connected"; }
    }
    join_thread(thread);
    delete_thread(thread);
}

#[test]
#[ignore]
fn test_http() {
    let thread = create_thread();
    start_thread(thread);
    let port = comm::port();
    let chan = comm::chan(port);
    connect(thread, 0u32, "74.125.224.146", chan);
    alt comm::recv(port) {
      connected(cd) {
        write(thread, 0u32, str::bytes("GET / HTTP/1.0\n\n"), chan);
        alt comm::recv(port) {
          wrote(cd) {
            read_start(thread, 0u32, chan);
            let keep_going = true;
            while keep_going {
                alt comm::recv(port) {
                  read(_, buf, -1) {
                    keep_going = false;
                    delete_buf(buf);
                  }
                  read(_, buf, len) {
                    unsafe {
                        log(error, len);
                        let buf = vec::unsafe::from_buf(buf,
                                                        len as uint);
                        let str = str::from_bytes(buf);
                        #error("read something");
                        io::println(str);
                    }
                    delete_buf(buf);
                  }
                  _ { fail "test_http: protocol error"; }
                }
            }
            close_connection(thread, 0u32);
          }
          _ { fail "test_http: expected `wrote`"; }
        }
      }
      _ { fail "test_http: port not connected"; }
    }
    join_thread(thread);
    delete_thread(thread);
}
