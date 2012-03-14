import map::hashmap;
export loop_new, loop_delete, run, close, run_in_bg;
export async_init, async_send;
export timer_init, timer_start, timer_stop;

// these are processed solely in the
// process_operation() crust fn below
enum uv_operation {
    op_async_init([u8]),
    op_close(uv_handle, *libc::c_void),
    op_timer_init([u8]),
    op_timer_start([u8], *libc::c_void, u32, u32),
    op_timer_stop([u8], *libc::c_void, fn~(uv_handle)),
    op_teardown(*libc::c_void)
}

enum uv_handle {
    uv_async([u8], uv_loop),
    uv_timer([u8], uv_loop)
}

enum uv_msg {
    // requests from library users
    msg_run(comm::chan<bool>),
    msg_run_in_bg(),
    msg_async_init(fn~(uv_handle), fn~(uv_handle)),
    msg_async_send([u8]),
    msg_close(uv_handle, fn~()),
    msg_timer_init(fn~(uv_handle)),
    msg_timer_start([u8], u32, u32, fn~(uv_handle)),
    msg_timer_stop([u8], fn~(uv_handle)),

    // dispatches from libuv
    uv_async_init([u8], *libc::c_void),
    uv_async_send([u8]),
    uv_close([u8]),
    uv_timer_init([u8], *libc::c_void),
    uv_timer_call([u8]),
    uv_timer_stop([u8], fn~(uv_handle)),
    uv_end(),
    uv_teardown_check()
}

type uv_loop_data = {
    operation_port: comm::port<uv_operation>,
    rust_loop_chan: comm::chan<uv_msg>
};

enum uv_loop {
    uv_loop_new(comm::chan<uv_msg>, *libc::c_void)
}

#[nolink]
native mod rustrt {
    fn rust_uv_loop_new() -> *libc::c_void;
    fn rust_uv_loop_delete(lp: *libc::c_void);
    fn rust_uv_loop_set_data(
        lp: *libc::c_void,
        data: *uv_loop_data);
    fn rust_uv_bind_op_cb(lp: *libc::c_void, cb: *u8)
        -> *libc::c_void;
    fn rust_uv_stop_op_cb(handle: *libc::c_void);
    fn rust_uv_run(loop_handle: *libc::c_void);
    fn rust_uv_close(handle: *libc::c_void, cb: *u8);
    fn rust_uv_close_async(handle: *libc::c_void);
    fn rust_uv_close_timer(handle: *libc::c_void);
    fn rust_uv_async_send(handle: *libc::c_void);
    fn rust_uv_async_init(
        loop_handle: *libc::c_void,
        cb: *u8,
        id: *u8) -> *libc::c_void;
    fn rust_uv_timer_init(
        loop_handle: *libc::c_void,
        cb: *u8,
        id: *u8) -> *libc::c_void;
    fn rust_uv_timer_start(
        timer_handle: *libc::c_void,
        timeout: libc::c_uint,
        repeat: libc::c_uint);
    fn rust_uv_timer_stop(handle: *libc::c_void);
}

// public functions
fn loop_new() -> uv_loop unsafe {
    let ret_recv_port: comm::port<uv_loop> =
        comm::port();
    let ret_recv_chan: comm::chan<uv_loop> =
        comm::chan(ret_recv_port);

    task::spawn_sched(task::manual_threads(1u)) {||
        // our beloved uv_loop_t ptr
        let loop_handle = rustrt::
            rust_uv_loop_new();

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
        let user_uv_loop = uv_loop_new(rust_loop_chan, loop_handle);
        comm::send(ret_recv_chan, copy(user_uv_loop));

        // create our "special" async handle that will
        // allow all operations against libuv to be
        // "buffered" in the operation_port, for processing
        // from the thread that libuv runs on
        let loop_data: uv_loop_data = {
            operation_port: operation_port,
            rust_loop_chan: rust_loop_chan
        };
        rustrt::rust_uv_loop_set_data(
            loop_handle,
            ptr::addr_of(loop_data)); // pass an opaque C-ptr
                                      // to libuv, this will be
                                      // in the process_operation
                                      // crust fn
        let op_handle = rustrt::rust_uv_bind_op_cb(
            loop_handle,
            process_operation);

        // all state goes here
        let handles: map::hashmap<[u8], *libc::c_void> =
            map::new_bytes_hash();
        let id_to_handle: map::hashmap<[u8], uv_handle> =
            map::new_bytes_hash();
        let after_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::new_bytes_hash();
        let close_callbacks: map::hashmap<[u8], fn~()> =
            map::new_bytes_hash();
        let async_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::new_bytes_hash();
        let timer_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::new_bytes_hash();

        // the main loop that this task blocks on.
        // should have the same lifetime as the C libuv
        // event loop.
        let mut keep_going = true;
        while keep_going {
            alt comm::recv(rust_loop_port) {
              msg_run(end_chan) {
                // start the libuv event loop
                // we'll also do a uv_async_send with
                // the operation handle to have the
                // loop process any pending operations
                // once its up and running
                task::spawn_sched(task::manual_threads(1u)) {||
                    // make sure we didn't start the loop
                    // without the user registering handles
                    comm::send(rust_loop_chan, uv_teardown_check);
                    // this call blocks
                    rustrt::rust_uv_run(loop_handle);
                    // when we're done, msg the
                    // end chan
                    comm::send(end_chan, true);
                    comm::send(rust_loop_chan, uv_end);
                };
              }

              msg_run_in_bg {
                task::spawn_sched(task::manual_threads(1u)) {||
                    // see note above
                    comm::send(rust_loop_chan, uv_teardown_check);
                    // this call blocks
                    rustrt::rust_uv_run(loop_handle);
                };
              }

              msg_close(handle, cb) {
                let id = get_id_from_handle(handle);
                close_callbacks.insert(id, cb);
                let handle_ptr = handles.get(id);
                let op = op_close(handle, handle_ptr);

                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_close(id) {
                handles.remove(id);
                let handle = id_to_handle.get(id);
                id_to_handle.remove(id);
                alt handle {
                  uv_async(id, _) {
                    async_cbs.remove(id);
                  }
                  uv_timer(id, _) {
                    timer_cbs.remove(id);
                  }
                  _ {
                    fail "unknown form of uv_handle encountered "
                        + "in uv_close handler";
                  }
                }
                let cb = close_callbacks.get(id);
                close_callbacks.remove(id);
                task::spawn {||
                    cb();
                };
                // ask the rust loop to check and see if there
                // are no more user-registered handles
                comm::send(rust_loop_chan, uv_teardown_check);
              }

              msg_async_init(callback, after_cb) {
                // create a new async handle
                // with the id as the handle's
                // data and save the callback for
                // invocation on msg_async_send
                let id = gen_handle_id();
                handles.insert(id, ptr::null());
                async_cbs.insert(id, callback);
                after_cbs.insert(id, after_cb);
                let op = op_async_init(id);
                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_async_init(id, async_handle) {
                // libuv created a handle, which is
                // passed back to us. save it and
                // then invoke the supplied callback
                // for after completion
                handles.insert(id, async_handle);
                let after_cb = after_cbs.get(id);
                after_cbs.remove(id);
                let async = uv_async(id, user_uv_loop);
                id_to_handle.insert(id, copy(async));
                task::spawn {||
                    after_cb(async);
                };
              }

              msg_async_send(id) {
                let async_handle = handles.get(id);
                do_send(async_handle);
              }
              uv_async_send(id) {
                let async_cb = async_cbs.get(id);
                task::spawn {||
                    let the_loop = user_uv_loop;
                    async_cb(uv_async(id, the_loop));
                };
              }

              msg_timer_init(after_cb) {
                let id = gen_handle_id();
                handles.insert(id, ptr::null());
                after_cbs.insert(id, after_cb);
                let op = op_timer_init(id);
                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_timer_init(id, handle) {
                handles.insert(id, handle);
                let after_cb = after_cbs.get(id);
                after_cbs.remove(id);
                let new_timer = uv_timer(id, user_uv_loop);
                id_to_handle.insert(id, copy(new_timer));
                task::spawn {||
                    after_cb(new_timer);
                };
              }

              uv_timer_call(id) {
                let cb = timer_cbs.get(id);
                let the_timer = id_to_handle.get(id);
                task::spawn {||
                    cb(the_timer);
                };
              }

              msg_timer_start(id, timeout, repeat, timer_call_cb) {
                timer_cbs.insert(id, timer_call_cb);
                let handle = handles.get(id);
                let op = op_timer_start(id, handle, timeout,
                                        repeat);
                pass_to_libuv(op_handle, operation_chan, op);
              }

              msg_timer_stop(id, after_cb) {
                let handle = handles.get(id);
                let op = op_timer_stop(id, handle, after_cb);
                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_timer_stop(id, after_cb) {
                let the_timer = id_to_handle.get(id);
                after_cb(the_timer);
              }

              uv_teardown_check() {
                // here we're checking if there are no user-registered
                // handles (and the loop is running), if this is the
                // case, then we need to unregister the op_handle via
                // a uv_close() call, thus allowing libuv to return
                // on its own.
                if (handles.size() == 0u) {
                    let op = op_teardown(op_handle);
                    pass_to_libuv(op_handle, operation_chan, op);
                }
              }

              uv_end() {
                keep_going = false;
              }

              _ { fail "unknown form of uv_msg received"; }
            }
        }
    };
    ret comm::recv(ret_recv_port);
}

fn loop_delete(lp: uv_loop) {
    let loop_ptr = get_loop_ptr_from_uv_loop(lp);
    rustrt::rust_uv_loop_delete(loop_ptr);
}

fn run(lp: uv_loop) {
    let end_port = comm::port::<bool>();
    let end_chan = comm::chan::<bool>(end_port);
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg_run(end_chan));
    comm::recv(end_port);
}

fn run_in_bg(lp: uv_loop) {
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg_run_in_bg);
}

fn async_init (
    lp: uv_loop,
    async_cb: fn~(uv_handle),
    after_cb: fn~(uv_handle)) {
    let mut msg = msg_async_init(async_cb, after_cb);
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg);
}

fn async_send(async: uv_handle) {
    alt async {
      uv_async(id, lp) {
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        comm::send(loop_chan, msg_async_send(id));
      }
      _ {
        fail "attempting to call async_send() with a" +
            " uv_async uv_handle";
      }
    }
}

fn close(h: uv_handle, cb: fn~()) {
    let loop_chan = get_loop_chan_from_handle(h);
    comm::send(loop_chan, msg_close(h, cb));
}

fn timer_init(lp: uv_loop, after_cb: fn~(uv_handle)) {
    let mut msg = msg_timer_init(after_cb);
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg);
}

fn timer_start(the_timer: uv_handle, timeout: u32, repeat:u32,
               timer_cb: fn~(uv_handle)) {
    alt the_timer {
      uv_timer(id, lp) {
        let mut msg = msg_timer_start(id, timeout, repeat, timer_cb);
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        comm::send(loop_chan, msg);
      }
      _ {
        fail "can only pass a uv_timer form of uv_handle to "+
             " uv::timer_start()";
      }
    }
}

fn timer_stop(the_timer: uv_handle, after_cb: fn~(uv_handle)) {
    alt the_timer {
      uv_timer(id, lp) {
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        let mut msg = msg_timer_stop(id, after_cb);
        comm::send(loop_chan, msg);
      }
      _ {
        fail "only uv_timer form is allowed in calls to "+
             " uv::timer_stop()";
      }
    }
}

// internal functions
fn pass_to_libuv(
        op_handle: *libc::c_void,
        operation_chan: comm::chan<uv_operation>,
        op: uv_operation) unsafe {
    comm::send(operation_chan, copy(op));
    do_send(op_handle);
}
fn do_send(h: *libc::c_void) {
    rustrt::rust_uv_async_send(h);
}
fn gen_handle_id() -> [u8] {
    ret rand::rng().gen_bytes(16u);
}
fn get_handle_id_from(buf: *u8) -> [u8] unsafe {
    ret vec::unsafe::from_buf(buf, 16u);
}

fn get_loop_chan_from_data(data: *uv_loop_data)
        -> comm::chan<uv_msg> unsafe {
    ret (*data).rust_loop_chan;
}

fn get_loop_chan_from_handle(handle: uv_handle)
    -> comm::chan<uv_msg> {
    alt handle {
      uv_async(id,lp) | uv_timer(id,lp) {
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        ret loop_chan;
      }
      _ {
        fail "unknown form of uv_handle for get_loop_chan_from "
             + " handle";
      }
    }
}

fn get_loop_ptr_from_uv_loop(lp: uv_loop) -> *libc::c_void {
    alt lp {
      uv_loop_new(loop_chan, loop_ptr) {
        ret loop_ptr;
      }
    }
}
fn get_loop_chan_from_uv_loop(lp: uv_loop) -> comm::chan<uv_msg> {
    alt lp {
      uv_loop_new(loop_chan, loop_ptr) {
        ret loop_chan;
      }
    }
}

fn get_id_from_handle(handle: uv_handle) -> [u8] {
    alt handle {
      uv_async(id,lp) | uv_timer(id,lp) {
        ret id;
      }
      _ {
        fail "unknown form of uv_handle for get_id_from handle";
      }
    }
}

// crust
crust fn process_operation(
        lp: *libc::c_void,
        data: *uv_loop_data) unsafe {
    let op_port = (*data).operation_port;
    let loop_chan = get_loop_chan_from_data(data);
    let mut op_pending = comm::peek(op_port);
    while(op_pending) {
        alt comm::recv(op_port) {
          op_async_init(id) {
            let id_ptr = vec::unsafe::to_ptr(id);
            let async_handle = rustrt::rust_uv_async_init(
                lp,
                process_async_send,
                id_ptr);
            comm::send(loop_chan, uv_async_init(
                id,
                async_handle));
          }
          op_close(handle, handle_ptr) {
            handle_op_close(handle, handle_ptr);
          }
          op_timer_init(id) {
            let id_ptr = vec::unsafe::to_ptr(id);
            let timer_handle = rustrt::rust_uv_timer_init(
                lp,
                process_timer_call,
                id_ptr);
            comm::send(loop_chan, uv_timer_init(
                id,
                timer_handle));
          }
          op_timer_start(id, handle, timeout, repeat) {
            rustrt::rust_uv_timer_start(handle, timeout,
                                              repeat);
          }
          op_timer_stop(id, handle, after_cb) {
            rustrt::rust_uv_timer_stop(handle);
            comm::send(loop_chan, uv_timer_stop(id, after_cb));
          }
          op_teardown(op_handle) {
            // this is the last msg that'll be processed by
            // this fn, in the current lifetime of the handle's
            // uv_loop_t
            rustrt::rust_uv_stop_op_cb(op_handle);
          }
          _ { fail "unknown form of uv_operation received"; }
        }
        op_pending = comm::peek(op_port);
    }
}

fn handle_op_close(handle: uv_handle, handle_ptr: *libc::c_void) {
    // it's just like im doing C
    alt handle {
      uv_async(id, lp) {
        let cb = process_close_async;
        rustrt::rust_uv_close(
            handle_ptr, cb);
      }
      uv_timer(id, lp) {
        let cb = process_close_timer;
        rustrt::rust_uv_close(
            handle_ptr, cb);
      }
      _ {
        fail "unknown form of uv_handle encountered " +
            "in process_operation/op_close";
      }
    }
}

crust fn process_async_send(id_buf: *u8, data: *uv_loop_data)
    unsafe {
    let handle_id = get_handle_id_from(id_buf);
    let loop_chan = get_loop_chan_from_data(data);
    comm::send(loop_chan, uv_async_send(handle_id));
}

crust fn process_timer_call(id_buf: *u8, data: *uv_loop_data)
    unsafe {
    let handle_id = get_handle_id_from(id_buf);
    let loop_chan = get_loop_chan_from_data(data);
    comm::send(loop_chan, uv_timer_call(handle_id));
}

fn process_close_common(id: [u8], data: *uv_loop_data)
    unsafe {
    // notify the rust loop that their handle is closed, then
    // the caller will invoke a per-handle-type c++ func to
    // free allocated memory
    let loop_chan = get_loop_chan_from_data(data);
    comm::send(loop_chan, uv_close(id));
}

crust fn process_close_async(
    id_buf: *u8,
    handle_ptr: *libc::c_void,
    data: *uv_loop_data)
    unsafe {
    let id = get_handle_id_from(id_buf);
    rustrt::rust_uv_close_async(handle_ptr);
    // at this point, the handle and its data has been
    // released. notify the rust loop to remove the
    // handle and its data and call the user-supplied
    // close cb
    process_close_common(id, data);
}

crust fn process_close_timer(
    id_buf: *u8,
    handle_ptr: *libc::c_void,
    data: *uv_loop_data)
    unsafe {
    let id = get_handle_id_from(id_buf);
    rustrt::rust_uv_close_timer(handle_ptr);
    process_close_common(id, data);
}


#[test]
fn test_uv_new_loop_no_handles() {
    let test_loop = uv::loop_new();
    uv::run(test_loop); // this should return immediately
                    // since there aren't any handles..
    uv::loop_delete(test_loop);
}

#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_simple_async() {
    let test_loop = uv::loop_new();
    let exit_port = comm::port::<bool>();
    let exit_chan = comm::chan::<bool>(exit_port);
    uv::async_init(test_loop, {|new_async|
        uv::close(new_async) {||
            comm::send(exit_chan, true);
        };
    }, {|new_async|
        uv::async_send(new_async);
    });
    uv::run(test_loop);
    let result = comm::recv(exit_port);
    assert result;
    uv::loop_delete(test_loop);
}

#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_timer() {
    let test_loop = uv::loop_new();
    let exit_port = comm::port::<bool>();
    let exit_chan = comm::chan::<bool>(exit_port);
    uv::timer_init(test_loop) {|new_timer|
        uv::timer_start(new_timer, 1u32, 0u32) {|started_timer|
            uv::timer_stop(started_timer) {|stopped_timer|
                uv::close(stopped_timer) {||
                    comm::send(exit_chan, true);
                };
            };
        };
    };
    uv::run(test_loop);
    assert comm::recv(exit_port);
    uv::loop_delete(test_loop);
}
