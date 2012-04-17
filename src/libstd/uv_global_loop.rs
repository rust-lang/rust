#[doc="
Process-wide, lazily started/stopped libuv event loop interaction.
"];

import ll = uv_ll;
import hl = uv_hl;
import get_gl = get;

export get;

native mod rustrt {
    fn rust_uv_get_kernel_global_chan_ptr() -> *libc::uintptr_t;
    fn rust_uv_get_kernel_global_async_handle() -> *libc::uintptr_t;
    fn rust_compare_and_swap_ptr(address: *libc::uintptr_t,
                                 oldval: libc::uintptr_t,
                                 newval: libc::uintptr_t) -> bool;
}

#[doc ="
Race-free helper to get access to a global task where a libuv
loop is running.

# Return

* A `hl::high_level_loop` that encapsulates communication with the global
loop.
"]
fn get() -> hl::high_level_loop {
    let global_loop_chan_ptr = rustrt::rust_uv_get_kernel_global_chan_ptr();
    log(debug, #fmt("ENTERING global_loop::get() loop chan: %?",
       global_loop_chan_ptr));

    let builder_fn = {||
        let builder = task::builder();
        let opts = {
            supervise: false,
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
        let chan = priv::chan_from_global_ptr::<hl::high_level_msg>(
            global_loop_chan_ptr,
            builder_fn) {|port|

            // the actual body of our global loop lives here
            log(debug, "initialized global port task!");
            log(debug, "GLOBAL initialized global port task!");
            outer_global_loop_body(port);
        };
        log(debug, "after priv::chan_from_global_ptr");
        let handle = get_global_async_handle_native_representation()
            as **ll::uv_async_t;
        ret { async_handle: handle, op_chan: chan };
    }
}

// INTERNAL API

unsafe fn outer_global_loop_body(msg_po: comm::port<hl::high_level_msg>) {
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
                    log(debug, #fmt("weak_exit_po recv'd msg: %?",
                                   left_val));
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
}

unsafe fn inner_global_loop_body(weak_exit_po_in: comm::port<()>,
                          msg_po_in: comm::port<hl::high_level_msg>,
                          loop_ptr: *libc::c_void,
                          -first_interaction: hl::high_level_msg) -> bool {
    // resend the msg
    comm::send(comm::chan(msg_po_in), first_interaction);

    // black magic
    let weak_exit_po_ptr = ptr::addr_of(weak_exit_po_in);
    hl::run_high_level_loop(
        loop_ptr,
        msg_po_in,
        // before_run
        {|async_handle|
            log(debug,#fmt("global_loop before_run: async_handle %?",
                          async_handle));
            // set the handle as the global
            set_global_async_handle(0u as *ll::uv_async_t,
                                    async_handle);
            // when this is ran, our async_handle is set up, so let's
            // do an async_send with it
            ll::async_send(async_handle);
        },
        // before_msg_drain
        {|async_handle|
            log(debug,#fmt("global_loop before_msg_drain: async_handle %?",
                          async_handle));
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
        {|async_handle|
            log(debug,#fmt("global_loop before_tear_down: async_handle %?",
                          async_handle));
            set_global_async_handle(async_handle,
                                    0 as *ll::uv_async_t);
        });
    // supposed to return a bool to indicate to the enclosing loop whether
    // it should continue or not..
    ret true;
}

unsafe fn get_global_async_handle_native_representation()
    -> *libc::uintptr_t {
    ret rustrt::rust_uv_get_kernel_global_async_handle();
}

unsafe fn get_global_async_handle() -> *ll::uv_async_t {
    ret (*get_global_async_handle_native_representation()) as *ll::uv_async_t;
}

unsafe fn set_global_async_handle(old: *ll::uv_async_t,
                           new_ptr: *ll::uv_async_t) {
    rustrt::rust_compare_and_swap_ptr(
        get_global_async_handle_native_representation(),
        old as libc::uintptr_t,
        new_ptr as libc::uintptr_t);
}

#[cfg(test)]
mod test {
    crust fn simple_timer_close_cb(timer_ptr: *ll::uv_timer_t) unsafe {
        let exit_ch_ptr = ll::get_data_for_uv_handle(
            timer_ptr as *libc::c_void) as *comm::chan<bool>;
        let exit_ch = *exit_ch_ptr;
        comm::send(exit_ch, true);
        log(debug, #fmt("EXIT_CH_PTR simple_timer_close_cb exit_ch_ptr: %?",
                       exit_ch_ptr));
    }
    crust fn simple_timer_cb(timer_ptr: *ll::uv_timer_t,
                             status: libc::c_int) unsafe {
        log(debug, "in simple timer cb");
        ll::timer_stop(timer_ptr);
        let hl_loop = get_gl();
        hl::interact(hl_loop) {|loop_ptr|
            log(debug, "closing timer");
            //ll::close(timer_ptr as *libc::c_void, simple_timer_close_cb);
            hl::unref_handle(hl_loop, timer_ptr, simple_timer_close_cb);
            log(debug, "about to deref exit_ch_ptr");
            log(debug, "after msg sent on deref'd exit_ch");
        };
        log(debug, "exiting simple timer cb");
    }

    fn impl_uv_hl_simple_timer(hl_loop: hl::high_level_loop) unsafe {
        let exit_po = comm::port::<bool>();
        let exit_ch = comm::chan(exit_po);
        let exit_ch_ptr = ptr::addr_of(exit_ch);
        log(debug, #fmt("EXIT_CH_PTR newly created exit_ch_ptr: %?",
                       exit_ch_ptr));
        let timer_handle = ll::timer_t();
        let timer_ptr = ptr::addr_of(timer_handle);
        hl::interact(hl_loop) {|loop_ptr|
            log(debug, "user code inside interact loop!!!");
            let init_status = ll::timer_init(loop_ptr, timer_ptr);
            if(init_status == 0i32) {
                hl::ref_handle(hl_loop, timer_ptr);
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
        log(debug, "global_loop timer test: msg recv on exit_po, done..");
    }
    #[test]
    #[ignore(cfg(target_os = "freebsd"))]
    fn test_uv_global_loop_high_level_global_timer() unsafe {
        let hl_loop = get_gl();
        task::spawn_sched(task::manual_threads(1u), {||
            impl_uv_hl_simple_timer(hl_loop);
        });
        impl_uv_hl_simple_timer(hl_loop);
    }
}