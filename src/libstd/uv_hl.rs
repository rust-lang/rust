#[doc = "
High-level bindings to work with the libuv library.

This module is geared towards library developers who want to
provide a high-level, abstracted interface to some set of
libuv functionality.
"];

export high_level_loop, hl_loop_ext, high_level_msg;
export run_high_level_loop, interact, ref, unref, unref_and_close;

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
    }),
    single_task_loop({
        async_handle: **ll::uv_async_t,
        op_chan: comm::chan<high_level_msg>
    }),
    monitor_task_loop({
        op_chan: comm::chan<high_level_msg>
    })
}

impl hl_loop_ext for high_level_loop {
    fn async_handle() -> **ll::uv_async_t {
        alt self {
          single_task_loop({async_handle, op_chan}) {
            ret async_handle;
          }
          _ {
            fail "variant of hl::high_level_loop that doesn't include" +
                "an async_handle field";
          }
        }
    }
    fn op_chan() -> comm::chan<high_level_msg> {
        alt self {
          single_task_loop({async_handle, op_chan}) {
            ret op_chan;
          }
          monitor_task_loop({op_chan}) {
            ret op_chan;
          }
          simple_task_loop({async_handle, op_chan}) {
            ret op_chan;
          }
        }
    }
}

#[doc="
Represents the range of interactions with a `high_level_loop`
"]
enum high_level_msg {
    interaction (fn~(*libc::c_void)),
    ref_handle (*libc::c_void),
    manual_unref_handle (*libc::c_void, option<*u8>),
    tear_down
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
                              before_msg_drain: fn~(*ll::uv_async_t) -> bool,
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
        before_msg_drain: before_msg_drain,
        before_tear_down: before_tear_down,
        msg_po_ptr: ptr::addr_of(msg_po),
        mut refd_handles: [mut],
        mut unrefd_handles: [mut]
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

iface uv_handle_manager<T> {
    fn init() -> T;
}

type safe_handle_fields<T> = {
    hl_loop: high_level_loop,
    handle: T,
    close_cb: *u8
};

/*fn safe_handle<T>(a_loop: high_level_loop,
                  handle_val: T,
                  handle_init_cb: fn~(*libc::c_void, *T),
                  close_cb: *u8) {

resource safe_handle_container<T>(handle_fields: safe_handle_fields<T>) {
}
}*/


#[doc="
Needs to be encapsulated within `safe_handle`
"]
fn ref<T>(hl_loop: high_level_loop, handle: *T) unsafe {
    send_high_level_msg(hl_loop, ref_handle(handle as *libc::c_void));
}
#[doc="
Needs to be encapsulated within `safe_handle`
"]
fn unref<T>(hl_loop: high_level_loop, handle: *T) unsafe {
    send_high_level_msg(hl_loop, manual_unref_handle(handle as *libc::c_void,
                                                   none));
}
fn unref_and_close<T>(hl_loop: high_level_loop, handle: *T, cb: *u8) unsafe {
    send_high_level_msg(hl_loop, manual_unref_handle(handle as *libc::c_void,
                                                   some(cb)));
}

// INTERNAL API

// data that lives for the lifetime of the high-evel oo
enum hl_loop_data {
    default_gl_data({
        async_handle: *ll::uv_async_t,
        mut active: bool,
        before_msg_drain: fn~(*ll::uv_async_t) -> bool,
        before_tear_down: fn~(*ll::uv_async_t),
        msg_po_ptr: *comm::port<high_level_msg>,
        mut refd_handles: [mut *libc::c_void],
        mut unrefd_handles: [mut *libc::c_void]})
}

unsafe fn send_high_level_msg(hl_loop: high_level_loop,
                              -msg: high_level_msg) unsafe {
    comm::send(hl_loop.op_chan(), msg);

    // if the global async handle == 0, then that means
    // the loop isn't active, so we don't need to wake it up,
    // (the loop's enclosing task should be blocking on a message
    // receive on this port)
    alt hl_loop {
      single_task_loop({async_handle, op_chan}) {
        if ((*async_handle) != 0 as *ll::uv_async_t) {
            log(debug,"global async handle != 0, waking up loop..");
            ll::async_send((*async_handle));
        }
        else {
            log(debug,"GLOBAL ASYNC handle == 0");
        }
      }
      simple_task_loop({async_handle, op_chan}) {
        log(debug,"simple async handle != 0, waking up loop..");
        ll::async_send((async_handle));
      }
      _ {}
    }
}

// this will be invoked by a call to uv::hl::interact() with
// the high_level_loop corresponding to this async_handle. We
// simply check if the loop is active and, if so, invoke the
// user-supplied on_wake callback that is stored in the loop's
// data member
crust fn high_level_wake_up_cb(async_handle: *ll::uv_async_t,
                               status: int) unsafe {
    // nothing here, yet.
    log(debug, #fmt("high_level_wake_up_cb crust.. handle: %? status: %?",
                     async_handle, status));
    let loop_ptr = ll::get_loop_for_uv_handle(async_handle);
    let data = ll::get_data_for_uv_handle(async_handle) as *hl_loop_data;
    // we check to see if the loop is "active" (the loop is set to
    // active = false the first time we realize we need to 'tear down',
    // set subsequent calls to the global async handle may be triggered
    // before all of the uv_close() calls are processed and loop exits
    // on its own. So if the loop isn't active, we won't run the user's
    // on_wake callback (and, consequently, let messages pile up, probably
    // in the loops msg_po)
    if (*data).active {
        log(debug, "before on_wake");
        let mut do_msg_drain = (*data).before_msg_drain(async_handle);
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
                      ref_handle(handle) {
                        high_level_ref(data, handle);
                      }
                      manual_unref_handle(handle, user_close_cb) {
                        high_level_unref(data, handle, true, user_close_cb);
                      }
                      tear_down {
                        log(debug,"incoming hl_msg: got tear_down");
                      }
                    }
                    continue = comm::peek(msg_po);
                }
            }
            else {
                log(debug, "in hl wake_cb, no pending messages");
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
    let data = ll::get_data_for_uv_handle(handle) as *hl_loop_data;
    if vec::len((*data).refd_handles) > 0u {
        fail "Didn't unref all high-level handles";
    }
}

fn high_level_tear_down(data: *hl_loop_data) unsafe {
    log(debug, "high_level_tear_down() called, close async_handle");
    // call user-suppled before_tear_down cb
    let async_handle = (*data).async_handle;
    (*data).before_tear_down(async_handle);
    ll::close(async_handle as *libc::c_void, tear_down_close_cb);
}

unsafe fn high_level_ref(data: *hl_loop_data, handle: *libc::c_void) {
    log(debug,"incoming hl_msg: got ..ref_handle");
    let mut refd_handles = (*data).refd_handles;
    let mut unrefd_handles = (*data).unrefd_handles;
    let handle_already_refd = refd_handles.contains(handle);
    if handle_already_refd {
        fail "attempt to do a high-level ref an already ref'd handle";
    }
    let handle_already_unrefd = unrefd_handles.contains(handle);
    // if we are ref'ing a handle (by ptr) that was already unref'd,
    // probably
    if handle_already_unrefd {
        let last_idx = vec::len(unrefd_handles) - 1u;
        let handle_idx = vec::position_elem(unrefd_handles, handle);
        alt handle_idx {
          none {
            fail "trying to remove handle that isn't in unrefd_handles";
          }
          some(idx) {
            unrefd_handles[idx] <-> unrefd_handles[last_idx];
            vec::pop(unrefd_handles);
          }
        }
        (*data).unrefd_handles = unrefd_handles;
    }
    refd_handles += [handle];
    (*data).refd_handles = refd_handles;
}

unsafe fn high_level_unref(data: *hl_loop_data, handle: *libc::c_void,
                   manual_unref: bool, user_close_cb: option<*u8>) {
    log(debug,"incoming hl_msg: got auto_unref_handle");
    let mut refd_handles = (*data).refd_handles;
    let mut unrefd_handles = (*data).unrefd_handles;
    log(debug, #fmt("refs: %?, unrefs %? handle %?", vec::len(refd_handles),
                    vec::len(unrefd_handles), handle));
    let handle_already_refd = refd_handles.contains(handle);
    if !handle_already_refd {
        fail "attempting to high-level unref an untracked handle";
    }
    let double_unref = unrefd_handles.contains(handle);
    if double_unref {
        log(debug, "double unref encountered");
        if manual_unref {
            // will allow a user to manual unref, but only signal
            // a fail when a double-unref is caused by a user
            fail "attempting to high-level unref an unrefd handle";
        }
        else {
            log(debug, "not failing...");
        }
    }
    else {
        log(debug, "attempting to unref handle");
        alt user_close_cb {
          some(cb) {
            ll::close(handle, cb);
          }
          none { }
        }
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
        unref_and_close(hl_loop, handle, async_close_cb);
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
            ref(hl_loop, ah_ptr);
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
                {|async_handle|
                    log(debug,#fmt("hltest before_msg_drain: async_handle %?",
                                  async_handle));
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
    #[ignore(cfg(target_os = "freebsd"))]
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
        let lifetime_handle = ll::async_t();
        let lifetime_handle_ptr = ptr::addr_of(lifetime_handle);
        interact(hl_loop) {|loop_ptr|
            ref(hl_loop, lifetime_handle_ptr);
            ll::async_init(loop_ptr, lifetime_handle_ptr,
                          lifetime_async_callback);
        };

        iter::repeat(7u) {||
            task::spawn_sched(task::manual_threads(1u), {||
                impl_uv_hl_async(hl_loop);
            });
        };
        impl_uv_hl_async(hl_loop);
        impl_uv_hl_async(hl_loop);
        impl_uv_hl_async(hl_loop);
        interact(hl_loop) {|loop_ptr|
            ll::close(lifetime_handle_ptr, lifetime_handle_close);
            unref(hl_loop, lifetime_handle_ptr);
            log(debug, "close and unref lifetime handle");
        };
        comm::recv(exit_po);
    }
}
