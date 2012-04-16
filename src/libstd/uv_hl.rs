#[doc = "
High-level bindings to work with the libuv library.

This module is geared towards library developers who want to
provide a high-level, abstracted interface to some set of
libuv functionality.
"];

export high_level_loop;
export run_high_level_loop, interact, ref_handle, unref_handle;
// this will eventually move into its own, unexported (from std) module
export get_global_loop;

import ll = uv_ll;

native mod rustrt {
    fn rust_uv_get_kernel_global_chan_ptr() -> *libc::uintptr_t;
    fn rust_uv_get_kernel_global_async_handle() -> **libc::c_void;
    fn rust_uv_set_kernel_global_async_handle(handle: *ll::uv_async_t);
    fn rust_uv_free_kernel_global_async_handle();
}

#[doc = "
Used to abstract-away direct interaction with a libuv loop.

# Arguments

* async_handle - a pointer to a pointer to a uv_async_t struct used to 'poke'
the C uv loop to process any pending callbacks

* op_chan - a channel used to send function callbacks to be processed
by the C uv loop
"]
type high_level_loop = {
    async_handle: **ll::uv_async_t,
    op_chan: comm::chan<high_level_msg>
};

#[doc = "
Race-free helper to get access to a global task where a libuv
loop is running.

# Return

* A `high_level_loop` that encapsulates communication with the global loop.
"]
fn get_global_loop() -> high_level_loop unsafe {
    let global_loop_chan_ptr = rustrt::rust_uv_get_kernel_global_chan_ptr();
    log(debug, #fmt("ENTERING get_global_loop() loop chan: %?",
       global_loop_chan_ptr));
    log(debug, #fmt("ENTERING get_global_loop() loop chan: %?",
       global_loop_chan_ptr));
    log(debug,#fmt("value of loop ptr: %?", *global_loop_chan_ptr));

    let builder_fn = {||
        let builder = task::builder();
        let opts = {
            supervise: true,
            notify_chan: none,
            sched:
                some({mode: task::manual_threads(1u),
                      native_stack_size: none })
        };
        task::set_opts(builder, opts);
        builder
    };
    unsafe {
        log(debug, "before priv::chan_from_global_ptr");
        let chan = priv::chan_from_global_ptr::<high_level_msg>(
            global_loop_chan_ptr,
            builder_fn) {|port|

            // the actual body of our global loop lives here
            log(debug, "initialized global port task!");
            log(debug, "GLOBAL!!!! initialized global port task!");
            outer_global_loop_body(port);
        };
        log(debug, "after priv::chan_from_global_ptr");
        let handle = get_global_async_handle();
        ret { async_handle: handle, op_chan: chan };
    }
}

#[doc = "
Takes a vanilla libuv `uv_loop_t*` ptr, performs some setup and then calls
`uv_run()`. Users will be able to access this loop via a provided
`async_handle` and `msg_ptr_po`. On top of libuv's internal handle refcount,
the high_level_loop manages its own lifetime with a similar refcount scheme.

This call blocks for the lifetime of the libuv loop.

# Arguments

* loop_ptr - a pointer to a currently unused libuv loop. Its `data` field
will be overwritten before the loop begins
* async_handle - a pointer to a _fresh_ `ll::uv_async_t` record that _has
not_ been initialized via `uv_async_init`, `ll::uv::async_init`, etc. It
must be a pointer to a clean rust `uv_async_t` record
* before_run - a unique closure that is invoked just before the call to
`uv_run`
* before_msg_drain - a unique closure that is invoked every time the loop is
awoken, but before the port pointed to in the `msg_po` argument is drained.
"]
unsafe fn run_high_level_loop(loop_ptr: *libc::c_void,
                              msg_po: comm::port<high_level_msg>,
                              before_run: fn~(*global_loop_data),
                              before_msg_drain: fn~() -> bool,
                              before_tear_down: fn~()) {
    // set up the special async handle we'll use to allow multi-task
    // communication with this loop
    let async = ll::async_t();
    let async_handle = ptr::addr_of(async);
    // associate the async handle with the loop
    ll::async_init(loop_ptr, async_handle, high_level_wake_up_cb);

    // initialize our loop data and store it in the loop
    let data: global_loop_data = {
        async_handle: async_handle,
        mut active: true,
        before_msg_drain: before_msg_drain,
        before_tear_down: before_tear_down,
        msg_po_ptr: ptr::addr_of(msg_po),
        mut refd_handles: [mut],
        mut unrefd_handles: [mut]
    };
    let data_ptr = ptr::addr_of(data);
    ll::set_data_for_uv_handle(async_handle, data_ptr);

    // call before_run
    before_run(data_ptr);

    log(debug, "about to run high level loop");
    // enter the loop... this blocks until the loop is done..
    ll::run(loop_ptr);
    log(debug, "high-level loop ended");
}

#[doc = "
Pass in a callback to be processed on the running libuv loop's thread

# Arguments

* a_loop - a high_level_loop record that represents a channel of
communication with an active libuv loop running on a thread
somwhere in the current process
* cb - a function callback to be processed on the running loop's
thread. The only parameter is an opaque pointer to the running
uv_loop_t. You can use this pointer to initiate or continue any
operations against the loop
"]
unsafe fn interact(a_loop: high_level_loop,
                      -cb: fn~(*libc::c_void)) {
    send_high_level_msg(a_loop, interaction(cb));
}

iface uv_handle_manager<T> {
    fn init() -> T;
}

resource uv_safe_handle<T>(handle_val: uv_handle_manager<T>) {
}

#[doc="
"]
fn ref_handle<T>(hl_loop: high_level_loop, handle: *T) unsafe {
    send_high_level_msg(hl_loop, auto_ref_handle(handle as *libc::c_void));
}
#[doc="
"]
fn unref_handle<T>(hl_loop: high_level_loop, handle: *T) unsafe {
    send_high_level_msg(hl_loop, auto_unref_handle(handle as *libc::c_void));
}

/////////////////////
// INTERNAL API
/////////////////////

unsafe fn send_high_level_msg(hl_loop: high_level_loop,
                              -msg: high_level_msg) unsafe {
    comm::send(hl_loop.op_chan, msg);

    // if the global async handle == 0, then that means
    // the loop isn't active, so we don't need to wake it up,
    // (the loop's enclosing task should be blocking on a message
    // receive on this port)
    if (*(hl_loop.async_handle) != 0 as *ll::uv_async_t) {
        log(debug,"global async handle != 0, waking up loop..");
        ll::async_send(*(hl_loop.async_handle));
    }
    else {
        log(debug,"GLOBAL ASYNC handle == 0");
    }
}

// this will be invoked by a call to uv::hl::interact() with
// the high_level_loop corresponding to this async_handle. We
// simply check if the loop is active and, if so, invoke the
// user-supplied on_wake callback that is stored in the loop's
// data member
crust fn high_level_wake_up_cb(async_handle: *libc::c_void,
                               status: int) unsafe {
    // nothing here, yet.
    log(debug, #fmt("high_level_wake_up_cb crust.. handle: %?",
                     async_handle));
    let loop_ptr = ll::get_loop_for_uv_handle(async_handle);
    let data = ll::get_data_for_uv_handle(async_handle) as *global_loop_data;
    // we check to see if the loop is "active" (the loop is set to
    // active = false the first time we realize we need to 'tear down',
    // set subsequent calls to the global async handle may be triggered
    // before all of the uv_close() calls are processed and loop exits
    // on its own. So if the loop isn't active, we won't run the user's
    // on_wake callback (and, consequently, let messages pile up, probably
    // in the loops msg_po)
    if (*data).active {
        log(debug, "before on_wake");
        let mut do_msg_drain = (*data).before_msg_drain();
        let mut continue = true;
        if do_msg_drain {
            let msg_po = *((*data).msg_po_ptr);
            if comm::peek(msg_po) {
                // if this is true, we'll iterate over the
                // msgs waiting in msg_po until there's no more
                log(debug,"got msg_po");
                while(continue) {
                    log(debug,"before alt'ing on high_level_msg");
                    alt comm::recv(msg_po) {
                      interaction(cb) {
                        log(debug,"got interaction, before cb..");
                        // call it..
                        cb(loop_ptr);
                        log(debug,"after calling cb");
                      }
                      auto_ref_handle(handle) {
                        high_level_ref(data, handle);
                      }
                      auto_unref_handle(handle) {
                        high_level_unref(data, handle, false);
                      }
                      tear_down {
                        log(debug,"incoming hl_msg: got tear_down");
                      }
                    }
                    continue = comm::peek(msg_po);
                }
            }
        }
        log(debug, #fmt("after on_wake, continue? %?", continue));
        if !do_msg_drain {
            high_level_tear_down(data);
        }
    }
}

crust fn tear_down_close_cb(handle: *ll::uv_async_t) unsafe {
    log(debug, #fmt("tear_down_close_cb called, closing handle at %?",
                    handle));
    // TODO: iterate through open handles on the loop and uv_close()
    // them all
    //let data = ll::get_data_for_uv_handle(handle) as *global_loop_data;
}

fn high_level_tear_down(data: *global_loop_data) unsafe {
    log(debug, "high_level_tear_down() called, close async_handle");
    // call user-suppled before_tear_down cb
    (*data).before_tear_down();
    let async_handle = (*data).async_handle;
    ll::close(async_handle as *libc::c_void, tear_down_close_cb);
}

unsafe fn high_level_ref(data: *global_loop_data, handle: *libc::c_void) {
    log(debug,"incoming hl_msg: got auto_ref_handle");
    let mut refd_handles = (*data).refd_handles;
    let handle_already_refd = refd_handles.contains(handle);
    if handle_already_refd {
        fail "attempt to do a high-level ref an already ref'd handle";
    }
    refd_handles += [handle];
    (*data).refd_handles = refd_handles;
}

crust fn auto_unref_close_cb(handle: *libc::c_void) {
    log(debug, "closing handle via high_level_unref");
}

unsafe fn high_level_unref(data: *global_loop_data, handle: *libc::c_void,
                   manual_unref: bool) {
    log(debug,"incoming hl_msg: got auto_unref_handle");
    let mut refd_handles = (*data).refd_handles;
    let mut unrefd_handles = (*data).unrefd_handles;
    let handle_already_refd = refd_handles.contains(handle);
    if !handle_already_refd {
        fail "attempting to high-level unref an untracked handle";
    }
    let double_unref = unrefd_handles.contains(handle);
    if double_unref {
        if manual_unref {
            // will allow a user to manual unref, but only signal
            // a fail when a double-unref is caused by a user
            fail "attempting to high-level unref an unrefd handle";
        }
    }
    else {
        ll::close(handle, auto_unref_close_cb);
        let last_idx = vec::len(refd_handles) - 1u;
        let handle_idx = vec::position_elem(refd_handles, handle);
        alt handle_idx {
          none {
            fail "trying to remove handle that isn't in refd_handles";
          }
          some(idx) {
            refd_handles[idx] <-> refd_handles[last_idx];
            vec::pop(refd_handles);
          }
        }
        (*data).refd_handles = refd_handles;
        unrefd_handles += [handle];
        (*data).unrefd_handles = unrefd_handles;
        if vec::len(refd_handles) == 0u {
            log(debug, "0 referenced handles, start loop teardown");
            high_level_tear_down(data);
        }
        else {
            log(debug, "more than 0 referenced handles");
        }
    }

}

enum high_level_msg {
    interaction (fn~(*libc::c_void)),
    auto_ref_handle (*libc::c_void),
    auto_unref_handle (*libc::c_void),
    tear_down
}

fn get_global_async_handle() -> **ll::uv_async_t {
    ret rustrt::rust_uv_get_kernel_global_async_handle() as **ll::uv_async_t;
}

fn set_global_async_handle(handle: *ll::uv_async_t) {
    rustrt::rust_uv_set_kernel_global_async_handle(handle);
}

type global_loop_data = {
    async_handle: *ll::uv_async_t,
    mut active: bool,
    before_msg_drain: fn~() -> bool,
    before_tear_down: fn~(),
    msg_po_ptr: *comm::port<high_level_msg>,
    mut refd_handles: [mut *libc::c_void],
    mut unrefd_handles: [mut *libc::c_void]
};

unsafe fn outer_global_loop_body(msg_po: comm::port<high_level_msg>) {
    // we're going to use a single libuv-generated loop ptr
    // for the duration of the process
    let loop_ptr = ll::loop_new();

    // data structure for loop goes here..

    // immediately weaken the task this is running in.
    priv::weaken_task() {|weak_exit_po|
        // when we first enter this loop, we're going
        // to wait on stand-by to receive a request to
        // fire-up the libuv loop
        let mut continue = true;
        while continue {
            log(debug, "in outer_loop...");
            continue = either::either(
                {|left_val|
                    // bail out..
                    // if we catch this msg at this point,
                    // we should just be able to exit because
                    // the loop isn't active
                    log(debug, "got msg on weak_exit_po in outer loop");
                    false
                }, {|right_val|
                    log(debug, "about to enter inner loop");
                    inner_global_loop_body(weak_exit_po, msg_po, loop_ptr,
                                          copy(right_val))
                }, comm::select2(weak_exit_po, msg_po));
            log(debug,#fmt("GLOBAL LOOP EXITED, WAITING TO RESTART? %?",
                       continue));
        }
    };

    ll::loop_delete(loop_ptr);
    // once we get here, show's over.
    rustrt::rust_uv_free_kernel_global_async_handle();
}

unsafe fn inner_global_loop_body(weak_exit_po_in: comm::port<()>,
                          msg_po_in: comm::port<high_level_msg>,
                          loop_ptr: *libc::c_void,
                          -first_interaction: high_level_msg) -> bool {
    // resend the msg
    comm::send(comm::chan(msg_po_in), first_interaction);

    // black magic
    let weak_exit_po_ptr = ptr::addr_of(weak_exit_po_in);
    run_high_level_loop(
        loop_ptr,
        msg_po_in,
        // before_run
        {|data|
            // set the handle as the global
            set_global_async_handle((*data).async_handle);
            // when this is ran, our async_handle is set up, so let's
            // do an async_send with it
            ll::async_send((*data).async_handle);
        },
        // before_msg_drain
        {||
            log(debug,"entering before_msg_drain for the global loop");
            let weak_exit_po = *weak_exit_po_ptr;
            if(comm::peek(weak_exit_po)) {
                // if this is true, immediately bail and return false, causing
                // the libuv loop to start tearing down
                log(debug,"got weak_exit meg inside libuv loop");
                comm::recv(weak_exit_po);
                false
            }
            // if no weak_exit_po msg is received, then we'll let the
            // loop continue
            else {
                true
            }
        },
        // before_tear_down
        {||
            set_global_async_handle(0 as *ll::uv_async_t);
        });
    // supposed to return a bool to indicate to the enclosing loop whether
    // it should continue or not..
    ret true;
}

#[cfg(test)]
mod test {
    crust fn simple_timer_close_cb(timer_ptr: *ll::uv_timer_t) unsafe {
        log(debug, "UNUSED...");
    }
    crust fn simple_timer_cb(timer_ptr: *ll::uv_timer_t,
                             status: libc::c_int) unsafe {
        log(debug, "in simple timer cb");
        let exit_ch_ptr = ll::get_data_for_uv_handle(
            timer_ptr as *libc::c_void) as *comm::chan<bool>;
        ll::timer_stop(timer_ptr);
        let hl_loop = get_global_loop();
        interact(hl_loop) {|loop_ptr|
            log(debug, "closing timer");
            //ll::close(timer_ptr as *libc::c_void, simple_timer_close_cb);
            unref_handle(hl_loop, timer_ptr);
            log(debug, "about to deref exit_ch_ptr");
            let exit_ch = *exit_ch_ptr;
            comm::send(exit_ch, true);
            log(debug, "after msg sent on deref'd exit_ch");
        };
        log(debug, "exiting simple timer cb");
    }

    fn impl_uv_hl_simple_timer(hl_loop: high_level_loop) unsafe {
        let exit_po = comm::port::<bool>();
        let exit_ch = comm::chan(exit_po);
        let exit_ch_ptr = ptr::addr_of(exit_ch);
        let timer_handle = ll::timer_t();
        let timer_ptr = ptr::addr_of(timer_handle);
        interact(hl_loop) {|loop_ptr|
            log(debug, "user code inside interact loop!!!");
            let init_status = ll::timer_init(loop_ptr, timer_ptr);
            if(init_status == 0i32) {
                ref_handle(hl_loop, timer_ptr);
                ll::set_data_for_uv_handle(
                    timer_ptr as *libc::c_void,
                    exit_ch_ptr as *libc::c_void);
                let start_status = ll::timer_start(timer_ptr, simple_timer_cb,
                                                   1u, 0u);
                if(start_status == 0i32) {
                }
                else {
                    fail "failure on ll::timer_start()";
                }
            }
            else {
                fail "failure on ll::timer_init()";
            }
        };
        comm::recv(exit_po);
        log(debug, "test_uv_hl_simple_timer: msg recv on exit_po, done..");
    }
    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_hl_high_level_global_timer() unsafe {
        let hl_loop = get_global_loop();
        impl_uv_hl_simple_timer(hl_loop);
        impl_uv_hl_simple_timer(hl_loop);
    }
}