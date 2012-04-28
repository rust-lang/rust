#[doc = "
High-level bindings to work with the libuv library.

This module is geared towards library developers who want to
provide a high-level, abstracted interface to some set of
libuv functionality.
"];

export high_level_loop, high_level_msg;
export run_high_level_loop, interact;

import ll = uv_ll;

#[doc = "
Used to abstract-away direct interaction with a libuv loop.

# Arguments

* async_handle - a pointer to a pointer to a uv_async_t struct used to 'poke'
the C uv loop to process any pending callbacks

* op_chan - a channel used to send function callbacks to be processed
by the C uv loop
"]
enum high_level_loop {
    simple_task_loop({
        async_handle: *ll::uv_async_t,
        op_chan: comm::chan<high_level_msg>
    })
}

#[doc="
Represents the range of interactions with a `high_level_loop`
"]
enum high_level_msg {
    interaction (fn~(*libc::c_void)),
    teardown_loop
}

#[doc = "
Given a vanilla `uv_loop_t*`

# Arguments

* loop_ptr - a pointer to a currently unused libuv loop. Its `data` field
will be overwritten before the loop begins
must be a pointer to a clean rust `uv_async_t` record
* msg_po - an active port that receives `high_level_msg`s
* before_run - a unique closure that is invoked after `uv_async_init` is
called on the `async_handle` passed into this callback, just before `uv_run`
is called on the provided `loop_ptr`
* before_msg_drain - a unique closure that is invoked every time the loop is
awoken, but before the port pointed to in the `msg_po` argument is drained
* before_tear_down - called just before the loop invokes `uv_close()` on the
provided `async_handle`. `uv_run` should return shortly after
"]
unsafe fn run_high_level_loop(loop_ptr: *libc::c_void,
                              msg_po: comm::port<high_level_msg>,
                              before_run: fn~(*ll::uv_async_t),
                              before_msg_process:
                                fn~(*ll::uv_async_t, bool) -> bool,
                              before_tear_down: fn~(*ll::uv_async_t)) {
    // set up the special async handle we'll use to allow multi-task
    // communication with this loop
    let async = ll::async_t();
    let async_handle = ptr::addr_of(async);
    // associate the async handle with the loop
    ll::async_init(loop_ptr, async_handle, high_level_wake_up_cb);

    // initialize our loop data and store it in the loop
    let data: hl_loop_data = default_gl_data({
        async_handle: async_handle,
        mut active: true,
        before_msg_process: before_msg_process,
        before_tear_down: before_tear_down,
        msg_po_ptr: ptr::addr_of(msg_po)
    });
    let data_ptr = ptr::addr_of(data);
    ll::set_data_for_uv_handle(async_handle, data_ptr);

    // call before_run
    before_run(async_handle);

    log(debug, "about to run high level loop");
    // enter the loop... this blocks until the loop is done..
    ll::run(loop_ptr);
    log(debug, "high-level loop ended");
}

#[doc = "
Provide a callback to be processed by `a_loop`

The primary way to do operations again a running `high_level_loop` that
doesn't involve creating a uv handle via `safe_handle`

# Arguments

* a_loop - a `high_level_loop` that you want to do operations against
* cb - a function callback to be processed on the running loop's
thread. The only parameter is an opaque pointer to the running
uv_loop_t. In the context of this callback, it is safe to use this pointer
to do various uv_* API calls. _DO NOT_ send this pointer out via ports/chans
"]
unsafe fn interact(a_loop: high_level_loop,
                      -cb: fn~(*libc::c_void)) {
    send_high_level_msg(a_loop, interaction(cb));
}

// INTERNAL API

// data that lives for the lifetime of the high-evel oo
enum hl_loop_data {
    default_gl_data({
        async_handle: *ll::uv_async_t,
        mut active: bool,
        before_msg_process: fn~(*ll::uv_async_t, bool) -> bool,
        before_tear_down: fn~(*ll::uv_async_t),
        msg_po_ptr: *comm::port<high_level_msg>})
}

unsafe fn send_high_level_msg(hl_loop: high_level_loop,
                              -msg: high_level_msg) {
    let op_chan = alt hl_loop{simple_task_loop({async_handle, op_chan}){
      op_chan}};
    comm::send(op_chan, msg);

    // if the global async handle == 0, then that means
    // the loop isn't active, so we don't need to wake it up,
    // (the loop's enclosing task should be blocking on a message
    // receive on this port)
    alt hl_loop {
      simple_task_loop({async_handle, op_chan}) {
        log(debug,"simple async handle != 0, waking up loop..");
        ll::async_send((async_handle));
      }
    }
}

// this will be invoked by a call to uv::hl::interact() with
// the high_level_loop corresponding to this async_handle. We
// simply check if the loop is active and, if so, invoke the
// user-supplied on_wake callback that is stored in the loop's
// data member
crust fn high_level_wake_up_cb(async_handle: *ll::uv_async_t,
                               status: int) unsafe {
    log(debug, #fmt("high_level_wake_up_cb crust.. handle: %? status: %?",
                     async_handle, status));
    let loop_ptr = ll::get_loop_for_uv_handle(async_handle);
    let data = ll::get_data_for_uv_handle(async_handle) as *hl_loop_data;
    alt (*data).active {
      true {
        let msg_po = *((*data).msg_po_ptr);
        alt comm::peek(msg_po) {
          true {
            loop {
                let msg = comm::recv(msg_po);
                alt (*data).active {
                  true {
                    alt msg {
                      interaction(cb) {
                        (*data).before_msg_process(async_handle,
                                                   (*data).active);
                        cb(loop_ptr);
                      }
                      teardown_loop {
                        begin_teardown(data);
                      }
                    }
                  }
                  false {
                    // drop msg ?
                  }
                }
                if !comm::peek(msg_po) { break; }
            }
          }
          false {
            // no pending msgs
          }
        }
      }
      false {
        // loop not active
      }
    }
}

crust fn tear_down_close_cb(handle: *ll::uv_async_t) unsafe {
    let loop_ptr = ll::get_loop_for_uv_handle(handle);
    let loop_refs = ll::loop_refcount(loop_ptr);
    log(debug, #fmt("tear_down_close_cb called, closing handle at %? refs %?",
                    handle, loop_refs));
    assert loop_refs == 1i32;
}

fn begin_teardown(data: *hl_loop_data) unsafe {
    log(debug, "high_level_tear_down() called, close async_handle");
    // call user-suppled before_tear_down cb
    let async_handle = (*data).async_handle;
    (*data).before_tear_down(async_handle);
    ll::close(async_handle as *libc::c_void, tear_down_close_cb);
}

#[cfg(test)]
mod test {
    crust fn async_close_cb(handle: *ll::uv_async_t) unsafe {
        log(debug, #fmt("async_close_cb handle %?", handle));
        let exit_ch = (*(ll::get_data_for_uv_handle(handle)
                        as *ah_data)).exit_ch;
        comm::send(exit_ch, ());
    }
    crust fn async_handle_cb(handle: *ll::uv_async_t, status: libc::c_int)
        unsafe {
        log(debug, #fmt("async_handle_cb handle %? status %?",handle,status));
        let hl_loop = (*(ll::get_data_for_uv_handle(handle)
                        as *ah_data)).hl_loop;
        ll::close(handle, async_close_cb);
    }
    type ah_data = {
        hl_loop: high_level_loop,
        exit_ch: comm::chan<()>
    };
    fn impl_uv_hl_async(hl_loop: high_level_loop) unsafe {
        let async_handle = ll::async_t();
        let ah_ptr = ptr::addr_of(async_handle);
        let exit_po = comm::port::<()>();
        let exit_ch = comm::chan(exit_po);
        let ah_data = {
            hl_loop: hl_loop,
            exit_ch: exit_ch
        };
        let ah_data_ptr = ptr::addr_of(ah_data);
        interact(hl_loop) {|loop_ptr|
            ll::async_init(loop_ptr, ah_ptr, async_handle_cb);
            ll::set_data_for_uv_handle(ah_ptr, ah_data_ptr as *libc::c_void);
            ll::async_send(ah_ptr);
        };
        comm::recv(exit_po);
    }

    // this fn documents the bear minimum neccesary to roll your own
    // high_level_loop
    unsafe fn spawn_test_loop(exit_ch: comm::chan<()>) -> high_level_loop {
        let hl_loop_port = comm::port::<high_level_loop>();
        let hl_loop_ch = comm::chan(hl_loop_port);
        task::spawn_sched(task::manual_threads(1u)) {||
            let loop_ptr = ll::loop_new();
            let msg_po = comm::port::<high_level_msg>();
            let msg_ch = comm::chan(msg_po);
            run_high_level_loop(
                loop_ptr,
                msg_po,
                // before_run
                {|async_handle|
                    log(debug,#fmt("hltest before_run: async_handle %?",
                                  async_handle));
                    // do an async_send with it
                    ll::async_send(async_handle);
                    comm::send(hl_loop_ch, simple_task_loop({
                       async_handle: async_handle,
                       op_chan: msg_ch
                    }));
                },
                // before_msg_drain
                {|async_handle, status|
                    log(debug,#fmt("hltest before_msg_drain: handle %? %?",
                                  async_handle, status));
                    true
                },
                // before_tear_down
                {|async_handle|
                    log(debug,#fmt("hl test_loop b4_tear_down: async %?",
                                  async_handle));
            });
            ll::loop_delete(loop_ptr);
            comm::send(exit_ch, ());
        };
        ret comm::recv(hl_loop_port);
    }

    crust fn lifetime_handle_close(handle: *libc::c_void) unsafe {
        log(debug, #fmt("lifetime_handle_close ptr %?", handle));
    }

    crust fn lifetime_async_callback(handle: *libc::c_void,
                                     status: libc::c_int) {
        log(debug, #fmt("lifetime_handle_close ptr %? status %?",
                        handle, status));
    }

    #[test]
    fn test_uv_hl_async() unsafe {
        let exit_po = comm::port::<()>();
        let exit_ch = comm::chan(exit_po);
        let hl_loop = spawn_test_loop(exit_ch);

        // using this handle to manage the lifetime of the high_level_loop,
        // as it will exit the first time one of the impl_uv_hl_async() is
        // cleaned up with no one ref'd handles on the loop (Which can happen
        // under race-condition type situations.. this ensures that the loop
        // lives until, at least, all of the impl_uv_hl_async() runs have been
        // called, at least.
        let work_exit_po = comm::port::<()>();
        let work_exit_ch = comm::chan(work_exit_po);
        iter::repeat(7u) {||
            task::spawn_sched(task::manual_threads(1u), {||
                impl_uv_hl_async(hl_loop);
                comm::send(work_exit_ch, ());
            });
        };
        iter::repeat(7u) {||
            comm::recv(work_exit_po);
        };
        log(debug, "sending teardown_loop msg..");
        // the teardown msg usually comes, in the case of the global loop,
        // as a result of receiving a msg on the weaken_task port. but,
        // anyone rolling their own high_level_loop can decide when to
        // send the msg. it's assert and barf, though, if all of your
        // handles aren't uv_close'd first
        alt hl_loop {
          simple_task_loop({async_handle, op_chan}) {
            comm::send(op_chan, teardown_loop);
            ll::async_send(async_handle);
          }
        }
        comm::recv(exit_po);
        log(debug, "after recv on exit_po.. exiting..");
    }
}
