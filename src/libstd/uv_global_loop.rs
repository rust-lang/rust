#[doc="
A process-wide libuv event loop for library use.
"];

import ll = uv_ll;
import hl = uv_hl;
import get_gl = get;

export get, get_single_task_gl, get_monitor_task_gl;

native mod rustrt {
    fn rust_uv_get_kernel_global_chan_ptr() -> *libc::uintptr_t;
    fn rust_uv_get_kernel_monitor_global_chan_ptr() -> *libc::uintptr_t;
    fn rust_uv_get_kernel_global_async_handle() -> *libc::uintptr_t;
    fn rust_compare_and_swap_ptr(address: *libc::uintptr_t,
                                 oldval: libc::uintptr_t,
                                 newval: libc::uintptr_t) -> bool;
}

#[doc ="
Race-free helper to get access to a global task where a libuv
loop is running.

Use `uv::hl::interact`, `uv::hl::ref_handle` and `uv::hl::unref_handle` to
do operations against the global loop that this function returns.

# Return

* A `hl::high_level_loop` that encapsulates communication with the global
loop.
"]
fn get() -> hl::high_level_loop {
    ret get_monitor_task_gl();
}

#[doc(hidden)]
fn get_monitor_task_gl() -> hl::high_level_loop {
    let monitor_loop_chan =
        rustrt::rust_uv_get_kernel_monitor_global_chan_ptr();
    ret spawn_global_weak_task(
        monitor_loop_chan,
        {|weak_exit_po, msg_po, loop_ptr, first_msg|
            log(debug, "monitor gl: entering inner loop");
            unsafe {
                monitor_task_loop_body(weak_exit_po, msg_po, loop_ptr,
                                       copy(first_msg))
            }
        },
        {|msg_ch|
            hl::monitor_task_loop({op_chan: msg_ch})
        });
}

#[doc(hidden)]
fn get_single_task_gl() -> hl::high_level_loop {
    let global_loop_chan_ptr = rustrt::rust_uv_get_kernel_global_chan_ptr();
    ret spawn_global_weak_task(
        global_loop_chan_ptr,
        {|weak_exit_po, msg_po, loop_ptr, first_msg|
            log(debug, "single-task gl: about to enter inner loop");
            unsafe {
                single_task_loop_body(weak_exit_po, msg_po, loop_ptr,
                                      copy(first_msg))
            }
        },
        {|msg_ch|
            log(debug, "after priv::chan_from_global_ptr");
            unsafe {
                let handle = get_global_async_handle_native_representation()
                    as **ll::uv_async_t;
                hl::single_task_loop(
                    { async_handle: handle, op_chan: msg_ch })
            }
        }
    );
}

// INTERNAL API

fn spawn_global_weak_task(
        global_loop_chan_ptr: *libc::uintptr_t,
        weak_task_body_cb: fn~(
            comm::port<()>,
            comm::port<hl::high_level_msg>,
            *libc::c_void,
            hl::high_level_msg) -> bool,
        after_task_spawn_cb: fn~(comm::chan<hl::high_level_msg>)
          -> hl::high_level_loop) -> hl::high_level_loop {
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
        let msg_ch = priv::chan_from_global_ptr::<hl::high_level_msg>(
            global_loop_chan_ptr,
            builder_fn) {|port|

            // the actual body of our global loop lives here
            log(debug, "initialized global port task!");
            log(debug, "GLOBAL initialized global port task!");
            outer_global_loop_body(port, weak_task_body_cb);
        };
        ret after_task_spawn_cb(msg_ch);
    }
}

unsafe fn outer_global_loop_body(
    msg_po: comm::port<hl::high_level_msg>,
    weak_task_body_cb: fn~(
        comm::port<()>,
        comm::port<hl::high_level_msg>,
        *libc::c_void,
        hl::high_level_msg) -> bool) {
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
                    weak_task_body_cb(weak_exit_po, msg_po, loop_ptr,
                                      right_val)
                }, comm::select2(weak_exit_po, msg_po));
            log(debug,#fmt("GLOBAL LOOP EXITED, WAITING TO RESTART? %?",
                       continue));
        }
    };

    ll::loop_delete(loop_ptr);
}
        
unsafe fn monitor_task_loop_body(weak_exit_po_in: comm::port<()>,
                          msg_po_in: comm::port<hl::high_level_msg>,
                          loop_ptr: *libc::c_void,
                          -first_interaction: hl::high_level_msg) -> bool {
    // resend the msg to be handled in the select2 loop below..
    comm::send(comm::chan(msg_po_in), first_interaction);

    // our async_handle
    let async_handle_po = comm::port::<*ll::uv_async_t>();
    let async_handle_ch = comm::chan(async_handle_po);

    // the msg_po that libuv will be receiving on..
    let loop_msg_po = comm::port::<hl::high_level_msg>();
    let loop_msg_po_ptr = ptr::addr_of(loop_msg_po);
    let loop_msg_ch = comm::chan(loop_msg_po);

    // the question of whether unsupervising this will even do any
    // good is there.. but since this'll go into blocking in libuv with
    // a quickness.. any errors that occur (including inside crust) will
    // be segfaults.. so yeah.
    task::spawn_sched(task::manual_threads(1u)) {||
        let loop_msg_po_in = *loop_msg_po_ptr;
        hl::run_high_level_loop(
            loop_ptr,
            loop_msg_po_in, // here the loop gets handed a different message
                            // port, as we'll be receiving all of the messages
                            // initially and then passing them on..
            // before_run
            {|async_handle|
                log(debug,#fmt("monitor gl: before_run: async_handle %?",
                              async_handle));
                // when this is ran, our async_handle is set up, so let's
                // do an async_send with it.. letting the loop know, once it
                // starts, that is has work
                ll::async_send(async_handle);
                comm::send(async_handle_ch, copy(async_handle));
            },
            // before_msg_drain
            {|async_handle|
                log(debug,#fmt("monitor gl: b4_msg_drain: async_handle %?",
                              async_handle));
                true
            },
            // before_tear_down
            {|async_handle|
                log(debug,#fmt("monitor gl: b4_tear_down: async_handle %?",
                              async_handle));
            });
    };

    // our loop is set up, so let's emit the handle back out to our users..
    let async_handle = comm::recv(async_handle_po);
    // supposed to return a bool to indicate to the enclosing loop whether
    // it should continue or not..
    let mut continue_inner_loop = true;
    let mut didnt_get_hl_bailout = true;
    while continue_inner_loop {
        log(debug, "monitor task inner loop.. about to block on select2");
        continue_inner_loop = either::either(
            {|left_val|
                // bail out..
                log(debug, #fmt("monitor inner weak_exit_po recv'd msg: %?",
                               left_val));
                // TODO: make loop bail out
                didnt_get_hl_bailout = false;
                false
            }, {|right_val|
                // wake up our inner loop and pass it a msg..
                comm::send(loop_msg_ch, copy(right_val));
                ll::async_send(async_handle);
                true
            }, comm::select2(weak_exit_po_in, msg_po_in)
        )
    }
    didnt_get_hl_bailout
}

unsafe fn single_task_loop_body(weak_exit_po_in: comm::port<()>,
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