#[doc = "
Rust bindings to libuv

This is the base-module for various levels of bindings to
the libuv library.

These modules are seeing heavy work, currently, and the final
API layout should not be inferred from its current form.

This base module currently contains a historical, rust-based
implementation of a few libuv operations that hews closely to
the patterns of the libuv C-API. It was used, mostly, to explore
some implementation details and will most likely be deprecated
in the near future.

The `ll` module contains low-level mappings for working directly
with the libuv C-API.

The `hl` module contains a set of tools library developers can
use for interacting with an active libuv loop. This modules's
API is meant to be used to write high-level,
rust-idiomatic abstractions for utilizes libuv's asynchronous IO
facilities.
"];

import map::hashmap;
export loop_new, loop_delete, run, close, run_in_bg;
export async_init, async_send;
export timer_init, timer_start, timer_stop;

import ll = uv_ll;
export ll;

import hl = uv_hl;
export hl;

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
    fn rust_uv_hilvl_close(handle: *libc::c_void, cb: *u8);
    fn rust_uv_hilvl_close_async(handle: *libc::c_void);
    fn rust_uv_hilvl_close_timer(handle: *libc::c_void);
    fn rust_uv_async_send(handle: *ll::uv_async_t);
    fn rust_uv_hilvl_async_init(
        loop_handle: *libc::c_void,
        cb: *u8,
        id: *u8) -> *libc::c_void;
    fn rust_uv_hilvl_timer_init(
        loop_handle: *libc::c_void,
        cb: *u8,
        id: *u8) -> *libc::c_void;
    fn rust_uv_hilvl_timer_start(
        timer_handle: *libc::c_void,
        timeout: libc::c_uint,
        repeat: libc::c_uint);
    fn rust_uv_timer_stop(handle: *ll::uv_timer_t) -> libc::c_int;
    fn rust_uv_free(ptr: *libc::c_void);
    // sizeof testing helpers
    fn rust_uv_helper_uv_tcp_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_connect_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_buf_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_write_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_err_t_size() -> libc::c_uint;
    fn rust_uv_helper_sockaddr_in_size() -> libc::c_uint;
    fn rust_uv_helper_uv_async_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_timer_t_size() -> libc::c_uint;
}


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
    uv_timer([u8], uv_loop),
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
            map::bytes_hash();
        let id_to_handle: map::hashmap<[u8], uv_handle> =
            map::bytes_hash();
        let after_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::bytes_hash();
        let close_callbacks: map::hashmap<[u8], fn~()> =
            map::bytes_hash();
        let async_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::bytes_hash();
        let timer_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::bytes_hash();

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
    let msg = msg_async_init(async_cb, after_cb);
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
    let msg = msg_timer_init(after_cb);
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg);
}

fn timer_start(the_timer: uv_handle, timeout: u32, repeat:u32,
               timer_cb: fn~(uv_handle)) {
    alt the_timer {
      uv_timer(id, lp) {
        let msg = msg_timer_start(id, timeout, repeat, timer_cb);
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
        let msg = msg_timer_stop(id, after_cb);
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
    rustrt::rust_uv_async_send(h as *ll::uv_async_t);
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
            let async_handle = rustrt::rust_uv_hilvl_async_init(
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
            let timer_handle = rustrt::rust_uv_hilvl_timer_init(
                lp,
                process_timer_call,
                id_ptr);
            comm::send(loop_chan, uv_timer_init(
                id,
                timer_handle));
          }
          op_timer_start(id, handle, timeout, repeat) {
            rustrt::rust_uv_hilvl_timer_start(handle, timeout,
                                              repeat);
          }
          op_timer_stop(id, handle, after_cb) {
            rustrt::rust_uv_timer_stop(handle as *ll::uv_timer_t);
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
        rustrt::rust_uv_hilvl_close(
            handle_ptr, cb);
      }
      uv_timer(id, lp) {
        let cb = process_close_timer;
        rustrt::rust_uv_hilvl_close(
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
    rustrt::rust_uv_hilvl_close_async(handle_ptr);
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
    rustrt::rust_uv_hilvl_close_timer(handle_ptr);
    process_close_common(id, data);
}

#[cfg(test)]
mod test {
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

    enum tcp_read_data {
        tcp_read_eof,
        tcp_read_more([u8]),
        tcp_read_error
    }

    type request_wrapper = {
        write_req: *ll::uv_write_t,
        req_buf: *[ll::uv_buf_t],
        read_chan: *comm::chan<str>
    };

    crust fn after_close_cb(handle: *libc::c_void) {
        log(debug, #fmt("after uv_close! handle ptr: %?",
                        handle));
    }

    crust fn on_alloc_cb(handle: *libc::c_void,
                         ++suggested_size: libc::size_t)
        -> ll::uv_buf_t unsafe {
        log(debug, "on_alloc_cb!");
        let char_ptr = ll::malloc_buf_base_of(suggested_size);
        log(debug, #fmt("on_alloc_cb h: %? char_ptr: %u sugsize: %u",
                         handle,
                         char_ptr as uint,
                         suggested_size as uint));
        ret ll::buf_init(char_ptr, suggested_size);
    }

    crust fn on_read_cb(stream: *ll::uv_stream_t,
                        nread: libc::ssize_t,
                        ++buf: ll::uv_buf_t) unsafe {
        log(debug, #fmt("CLIENT entering on_read_cb nred: %d", nread));
        if (nread > 0) {
            // we have data
            log(debug, #fmt("CLIENT read: data! nread: %d", nread));
            ll::read_stop(stream);
            let client_data = ll::
                get_data_for_uv_handle(stream as *libc::c_void)
                  as *request_wrapper;
            let buf_base = ll::get_base_from_buf(buf);
            let buf_len = ll::get_len_from_buf(buf);
            let bytes = vec::unsafe::from_buf(buf_base, buf_len);
            let read_chan = *((*client_data).read_chan);
            let msg_from_server = str::from_bytes(bytes);
            comm::send(read_chan, msg_from_server);
            ll::close(stream as *libc::c_void, after_close_cb)
        }
        else if (nread == -1) {
            // err .. possibly EOF
            log(debug, "read: eof!");
        }
        else {
            // nread == 0 .. do nothing, just free buf as below
            log(debug, "read: do nothing!");
        }
        // when we're done
        ll::free_base_of_buf(buf);
        log(debug, "CLIENT exiting on_read_cb");
    }

    crust fn on_write_complete_cb(write_req: *ll::uv_write_t,
                                  status: libc::c_int) unsafe {
        log(debug, #fmt("CLIENT beginning on_write_complete_cb status: %d",
                         status as int));
        let stream = ll::get_stream_handle_from_write_req(write_req);
        log(debug, #fmt("CLIENT on_write_complete_cb: tcp:%d write_handle:%d",
            stream as int, write_req as int));
        let result = ll::read_start(stream, on_alloc_cb, on_read_cb);
        log(debug, #fmt("CLIENT ending on_write_complete_cb .. status: %d",
                         result as int));
    }

    crust fn on_connect_cb(connect_req_ptr: *ll::uv_connect_t,
                                 status: libc::c_int) unsafe {
        log(debug, #fmt("beginning on_connect_cb .. status: %d",
                         status as int));
        let stream =
            ll::get_stream_handle_from_connect_req(connect_req_ptr);
        if (status == 0i32) {
            log(debug, "on_connect_cb: in status=0 if..");
            let client_data = ll::get_data_for_req(
                connect_req_ptr as *libc::c_void)
                as *request_wrapper;
            let write_handle = (*client_data).write_req as *libc::c_void;
            log(debug, #fmt("on_connect_cb: tcp: %d write_hdl: %d",
                            stream as int, write_handle as int));
            let write_result = ll::write(write_handle,
                              stream as *libc::c_void,
                              (*client_data).req_buf,
                              on_write_complete_cb);
            log(debug, #fmt("on_connect_cb: ll::write() status: %d",
                             write_result as int));
        }
        else {
            let test_loop = ll::get_loop_for_uv_handle(
                stream as *libc::c_void);
            let err_msg = ll::get_last_err_info(test_loop);
            log(debug, err_msg);
            assert false;
        }
        log(debug, "finishing on_connect_cb");
    }

    fn impl_uv_tcp_request(ip: str, port: int, req_str: str,
                          client_chan: *comm::chan<str>) unsafe {
        let test_loop = ll::loop_new();
        let tcp_handle = ll::tcp_t();
        let tcp_handle_ptr = ptr::addr_of(tcp_handle);
        let connect_handle = ll::connect_t();
        let connect_req_ptr = ptr::addr_of(connect_handle);

        // this is the persistent payload of data that we
        // need to pass around to get this example to work.
        // In C, this would be a malloc'd or stack-allocated
        // struct that we'd cast to a void* and store as the
        // data field in our uv_connect_t struct
        let req_str_bytes = str::bytes(req_str);
        let req_msg_ptr: *u8 = vec::unsafe::to_ptr(req_str_bytes);
        log(debug, #fmt("req_msg ptr: %u", req_msg_ptr as uint));
        let req_msg = [
            ll::buf_init(req_msg_ptr, vec::len(req_str_bytes))
        ];
        // this is the enclosing record, we'll pass a ptr to
        // this to C..
        let write_handle = ll::write_t();
        let write_handle_ptr = ptr::addr_of(write_handle);
        log(debug, #fmt("tcp req: tcp stream: %d write_handle: %d",
                         tcp_handle_ptr as int,
                         write_handle_ptr as int));
        let client_data = { writer_handle: write_handle_ptr,
                    req_buf: ptr::addr_of(req_msg),
                    read_chan: client_chan };

        let tcp_init_result = ll::tcp_init(
            test_loop as *libc::c_void, tcp_handle_ptr);
        if (tcp_init_result == 0i32) {
            log(debug, "sucessful tcp_init_result");

            log(debug, "building addr...");
            let addr = ll::ip4_addr(ip, port);
            // FIXME ref #2064
            let addr_ptr = ptr::addr_of(addr);
            log(debug, #fmt("after build addr in rust. port: %u",
                             addr.sin_port as uint));

            // this should set up the connection request..
            log(debug, #fmt("b4 call tcp_connect connect cb: %u ",
                            on_connect_cb as uint));
            let tcp_connect_result = ll::tcp_connect(
                connect_req_ptr, tcp_handle_ptr,
                addr_ptr, on_connect_cb);
            if (tcp_connect_result == 0i32) {
                // not set the data on the connect_req
                // until its initialized
                ll::set_data_for_req(
                    connect_req_ptr as *libc::c_void,
                    ptr::addr_of(client_data) as *libc::c_void);
                ll::set_data_for_uv_handle(
                    tcp_handle_ptr as *libc::c_void,
                    ptr::addr_of(client_data) as *libc::c_void);
                log(debug, "before run tcp req loop");
                ll::run(test_loop);
                log(debug, "after run tcp req loop");
            }
            else {
               log(debug, "ll::tcp_connect() failure");
               assert false;
            }
        }
        else {
            log(debug, "ll::tcp_init() failure");
            assert false;
        }
        ll::loop_delete(test_loop);

    }

    crust fn server_after_close_cb(handle: *libc::c_void) unsafe {
        log(debug, #fmt("SERVER server stream closed, should exit.. h: %?",
                   handle));
    }

    crust fn client_stream_after_close_cb(handle: *libc::c_void)
        unsafe {
        log(debug, "SERVER: closed client stream, now closing server stream");
        let client_data = ll::get_data_for_uv_handle(
            handle) as
            *tcp_server_data;
        ll::close((*client_data).server as *libc::c_void,
                      server_after_close_cb);
    }

    crust fn after_server_resp_write(req: *ll::uv_write_t) unsafe {
        let client_stream_ptr =
            ll::get_stream_handle_from_write_req(req);
        log(debug, "SERVER: resp sent... closing client stream");
        ll::close(client_stream_ptr as *libc::c_void,
                      client_stream_after_close_cb)
    }

    crust fn on_server_read_cb(client_stream_ptr: *ll::uv_stream_t,
                               nread: libc::ssize_t,
                               ++buf: ll::uv_buf_t) unsafe {
        if (nread > 0) {
            // we have data
            log(debug, #fmt("SERVER read: data! nread: %d", nread));

            // pull out the contents of the write from the client
            let buf_base = ll::get_base_from_buf(buf);
            let buf_len = ll::get_len_from_buf(buf);
            log(debug, #fmt("SERVER buf base: %u, len: %u, nread: %d",
                             buf_base as uint,
                             buf_len as uint,
                             nread));
            let bytes = vec::unsafe::from_buf(buf_base, buf_len);
            let request_str = str::from_bytes(bytes);

            let client_data = ll::get_data_for_uv_handle(
                client_stream_ptr as *libc::c_void) as *tcp_server_data;

            let server_kill_msg = (*client_data).server_kill_msg;
            let write_req = (*client_data).server_write_req;
            if (str::contains(request_str, server_kill_msg)) {
                log(debug, "SERVER: client req contains kill_msg!");
                log(debug, "SERVER: sending response to client");
                ll::read_stop(client_stream_ptr);
                let server_chan = *((*client_data).server_chan);
                comm::send(server_chan, request_str);
                let write_result = ll::write(
                    write_req as *libc::c_void,
                    client_stream_ptr as *libc::c_void,
                    (*client_data).server_resp_buf,
                    after_server_resp_write);
                log(debug, #fmt("SERVER: resp write result: %d",
                            write_result as int));
                if (write_result != 0i32) {
                    log(debug, "bad result for server resp ll::write()");
                    log(debug, ll::get_last_err_info(
                        ll::get_loop_for_uv_handle(client_stream_ptr
                            as *libc::c_void)));
                    assert false;
                }
            }
            else {
                log(debug, "SERVER: client req !contain kill_msg!");
            }
        }
        else if (nread == -1) {
            // err .. possibly EOF
            log(debug, "read: eof!");
        }
        else {
            // nread == 0 .. do nothing, just free buf as below
            log(debug, "read: do nothing!");
        }
        // when we're done
        ll::free_base_of_buf(buf);
        log(debug, "SERVER exiting on_read_cb");
    }

    crust fn server_connection_cb(server_stream_ptr:
                                    *ll::uv_stream_t,
                                  status: libc::c_int) unsafe {
        log(debug, "client connecting!");
        let test_loop = ll::get_loop_for_uv_handle(
                               server_stream_ptr as *libc::c_void);
        if status != 0i32 {
            let err_msg = ll::get_last_err_info(test_loop);
            log(debug, #fmt("server_connect_cb: non-zero status: %?",
                         err_msg));
            ret;
        }
        let server_data = ll::get_data_for_uv_handle(
            server_stream_ptr as *libc::c_void) as *tcp_server_data;
        let client_stream_ptr = (*server_data).client;
        let client_init_result = ll::tcp_init(test_loop,
                                                  client_stream_ptr);
        ll::set_data_for_uv_handle(
            client_stream_ptr as *libc::c_void,
            server_data as *libc::c_void);
        if (client_init_result == 0i32) {
            log(debug, "successfully initialized client stream");
            let accept_result = ll::accept(server_stream_ptr as
                                                 *libc::c_void,
                                               client_stream_ptr as
                                                 *libc::c_void);
            if (accept_result == 0i32) {
                // start reading
                let read_result = ll::read_start(
                    client_stream_ptr as *ll::uv_stream_t,
                                                     on_alloc_cb,
                                                     on_server_read_cb);
                if (read_result == 0i32) {
                    log(debug, "successful server read start");
                }
                else {
                    log(debug, #fmt("server_connection_cb: bad read:%d",
                                    read_result as int));
                    assert false;
                }
            }
            else {
                log(debug, #fmt("server_connection_cb: bad accept: %d",
                            accept_result as int));
                assert false;
            }
        }
        else {
            log(debug, #fmt("server_connection_cb: bad client init: %d",
                        client_init_result as int));
            assert false;
        }
    }

    type tcp_server_data = {
        client: *ll::uv_tcp_t,
        server: *ll::uv_tcp_t,
        server_kill_msg: str,
        server_resp_buf: *[ll::uv_buf_t],
        server_chan: *comm::chan<str>,
        server_write_req: *ll::uv_write_t
    };

    type async_handle_data = {
        continue_chan: *comm::chan<bool>
    };

    crust fn async_close_cb(handle: *libc::c_void) {
        log(debug, #fmt("SERVER: closing async cb... h: %?",
                   handle));
    }

    crust fn continue_async_cb(async_handle: *ll::uv_async_t,
                               status: libc::c_int) unsafe {
        // once we're in the body of this callback,
        // the tcp server's loop is set up, so we
        // can continue on to let the tcp client
        // do its thang
        let data = ll::get_data_for_uv_handle(
            async_handle as *libc::c_void) as *async_handle_data;
        let continue_chan = *((*data).continue_chan);
        let should_continue = status == 0i32;
        comm::send(continue_chan, should_continue);
        ll::close(async_handle as *libc::c_void, async_close_cb);
    }

    fn impl_uv_tcp_server(server_ip: str,
                          server_port: int,
                          kill_server_msg: str,
                          server_resp_msg: str,
                          server_chan: *comm::chan<str>,
                          continue_chan: *comm::chan<bool>) unsafe {
        let test_loop = ll::loop_new();
        let tcp_server = ll::tcp_t();
        let tcp_server_ptr = ptr::addr_of(tcp_server);

        let tcp_client = ll::tcp_t();
        let tcp_client_ptr = ptr::addr_of(tcp_client);

        let server_write_req = ll::write_t();
        let server_write_req_ptr = ptr::addr_of(server_write_req);

        let resp_str_bytes = str::bytes(server_resp_msg);
        let resp_msg_ptr: *u8 = vec::unsafe::to_ptr(resp_str_bytes);
        log(debug, #fmt("resp_msg ptr: %u", resp_msg_ptr as uint));
        let resp_msg = [
            ll::buf_init(resp_msg_ptr, vec::len(resp_str_bytes))
        ];

        let continue_async_handle = ll::async_t();
        let continue_async_handle_ptr =
            ptr::addr_of(continue_async_handle);
        let async_data =
            { continue_chan: continue_chan };
        let async_data_ptr = ptr::addr_of(async_data);

        let server_data: tcp_server_data = {
            client: tcp_client_ptr,
            server: tcp_server_ptr,
            server_kill_msg: kill_server_msg,
            server_resp_buf: ptr::addr_of(resp_msg),
            server_chan: server_chan,
            server_write_req: server_write_req_ptr
        };
        let server_data_ptr = ptr::addr_of(server_data);
        ll::set_data_for_uv_handle(tcp_server_ptr as *libc::c_void,
                                       server_data_ptr as *libc::c_void);

        // uv_tcp_init()
        let tcp_init_result = ll::tcp_init(
            test_loop as *libc::c_void, tcp_server_ptr);
        if (tcp_init_result == 0i32) {
            let server_addr = ll::ip4_addr(server_ip, server_port);
            // FIXME ref #2064
            let server_addr_ptr = ptr::addr_of(server_addr);

            // uv_tcp_bind()
            let bind_result = ll::tcp_bind(tcp_server_ptr,
                                               server_addr_ptr);
            if (bind_result == 0i32) {
                log(debug, "successful uv_tcp_bind, listening");

                // uv_listen()
                let listen_result = ll::listen(tcp_server_ptr as
                                                     *libc::c_void,
                                                   128i32,
                                                   server_connection_cb);
                if (listen_result == 0i32) {
                    // let the test know it can set up the tcp server,
                    // now.. this may still present a race, not sure..
                    let async_result = ll::async_init(test_loop,
                                       continue_async_handle_ptr,
                                       continue_async_cb);
                    if (async_result == 0i32) {
                        ll::set_data_for_uv_handle(
                            continue_async_handle_ptr as *libc::c_void,
                            async_data_ptr as *libc::c_void);
                        ll::async_send(continue_async_handle_ptr);
                        // uv_run()
                        ll::run(test_loop);
                        log(debug, "server uv::run() has returned");
                    }
                    else {
                        log(debug, #fmt("uv_async_init failure: %d",
                                async_result as int));
                        assert false;
                    }
                }
                else {
                    log(debug, #fmt("non-zero result on uv_listen: %d",
                                listen_result as int));
                    assert false;
                }
            }
            else {
                log(debug, #fmt("non-zero result on uv_tcp_bind: %d",
                            bind_result as int));
                assert false;
            }
        }
        else {
            log(debug, #fmt("non-zero result on uv_tcp_init: %d",
                        tcp_init_result as int));
            assert false;
        }
        ll::loop_delete(test_loop);
    }

    // this is the impl for a test that is (maybe) ran on a
    // per-platform/arch basis below
    fn impl_uv_tcp_server_and_request() unsafe {
        let bind_ip = "0.0.0.0";
        let request_ip = "127.0.0.1";
        let port = 8888;
        let kill_server_msg = "does a dog have buddha nature?";
        let server_resp_msg = "mu!";
        let client_port = comm::port::<str>();
        let client_chan = comm::chan::<str>(client_port);
        let server_port = comm::port::<str>();
        let server_chan = comm::chan::<str>(server_port);

        let continue_port = comm::port::<bool>();
        let continue_chan = comm::chan::<bool>(continue_port);
        let continue_chan_ptr = ptr::addr_of(continue_chan);

        task::spawn_sched(task::manual_threads(1u)) {||
            impl_uv_tcp_server(bind_ip, port,
                               kill_server_msg,
                               server_resp_msg,
                               ptr::addr_of(server_chan),
                               continue_chan_ptr);
        };

        // block until the server up is.. possibly a race?
        log(debug, "before receiving on server continue_port");
        comm::recv(continue_port);
        log(debug, "received on continue port, set up tcp client");

        task::spawn_sched(task::manual_threads(1u)) {||
            impl_uv_tcp_request(request_ip, port,
                               kill_server_msg,
                               ptr::addr_of(client_chan));
        };

        let msg_from_client = comm::recv(server_port);
        let msg_from_server = comm::recv(client_port);

        assert str::contains(msg_from_client, kill_server_msg);
        assert str::contains(msg_from_server, server_resp_msg);
    }

    // don't run this test on fbsd or 32bit linux
    #[cfg(target_os="win32")]
    #[cfg(target_os="darwin")]
    #[cfg(target_os="linux")]
    mod tcp_and_server_client_test {
        #[cfg(target_arch="x86_64")]
        mod impl64 {
            #[test]
            fn test_uv_tcp_server_and_request() unsafe {
                impl_uv_tcp_server_and_request();
            }
        }
        #[cfg(target_arch="x86")]
        mod impl32 {
            #[test]
            #[ignore(cfg(target_os = "linux"))]
            fn test_uv_tcp_server_and_request() unsafe {
                impl_uv_tcp_server_and_request();
            }
        }
    }

    // struct size tests
    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_struct_size_uv_tcp_t() {
        let native_handle_size = rustrt::rust_uv_helper_uv_tcp_t_size();
        let rust_handle_size = sys::size_of::<ll::uv_tcp_t>();
        let output = #fmt("uv_tcp_t -- native: %u rust: %u",
                          native_handle_size as uint, rust_handle_size);
        log(debug, output);
        assert native_handle_size as uint == rust_handle_size;
    }
    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_struct_size_uv_connect_t() {
        let native_handle_size =
            rustrt::rust_uv_helper_uv_connect_t_size();
        let rust_handle_size = sys::size_of::<ll::uv_connect_t>();
        let output = #fmt("uv_connect_t -- native: %u rust: %u",
                          native_handle_size as uint, rust_handle_size);
        log(debug, output);
        assert native_handle_size as uint == rust_handle_size;
    }
    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_struct_size_uv_buf_t() {
        let native_handle_size =
            rustrt::rust_uv_helper_uv_buf_t_size();
        let rust_handle_size = sys::size_of::<ll::uv_buf_t>();
        let output = #fmt("uv_buf_t -- native: %u rust: %u",
                          native_handle_size as uint, rust_handle_size);
        log(debug, output);
        assert native_handle_size as uint == rust_handle_size;
    }
    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_struct_size_uv_write_t() {
        let native_handle_size =
            rustrt::rust_uv_helper_uv_write_t_size();
        let rust_handle_size = sys::size_of::<ll::uv_write_t>();
        let output = #fmt("uv_write_t -- native: %u rust: %u",
                          native_handle_size as uint, rust_handle_size);
        log(debug, output);
        assert native_handle_size as uint == rust_handle_size;
    }

    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_struct_size_sockaddr_in() {
        let native_handle_size =
            rustrt::rust_uv_helper_sockaddr_in_size();
        let rust_handle_size = sys::size_of::<ll::sockaddr_in>();
        let output = #fmt("sockaddr_in -- native: %u rust: %u",
                          native_handle_size as uint, rust_handle_size);
        log(debug, output);
        assert native_handle_size as uint == rust_handle_size;
    }

    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_struct_size_uv_async_t() {
        let native_handle_size =
            rustrt::rust_uv_helper_uv_async_t_size();
        let rust_handle_size = sys::size_of::<ll::uv_async_t>();
        let output = #fmt("uv_async_t -- native: %u rust: %u",
                          native_handle_size as uint, rust_handle_size);
        log(debug, output);
        assert native_handle_size as uint == rust_handle_size;
    }
    
    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_struct_size_uv_timer_t() {
        let native_handle_size =
            rustrt::rust_uv_helper_uv_timer_t_size();
        let rust_handle_size = sys::size_of::<ll::uv_timer_t>();
        let output = #fmt("uv_timer_t -- native: %u rust: %u",
                          native_handle_size as uint, rust_handle_size);
        log(debug, output);
        assert native_handle_size as uint == rust_handle_size;
    }

}
