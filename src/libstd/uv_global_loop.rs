#[doc="
A process-wide libuv event loop for library use.
"];

import ll = uv_ll;
import hl = uv_hl;
import get_gl = get;

export get, get_monitor_task_gl;

native mod rustrt {
    fn rust_uv_get_kernel_global_chan_ptr() -> *libc::uintptr_t;
}

#[doc ="
Race-free helper to get access to a global task where a libuv
loop is running.

Use `uv::hl::interact` to do operations against the global
loop that this function returns.

# Return

* A `hl::high_level_loop` that encapsulates communication with the global
loop.
"]
fn get() -> hl::high_level_loop {
    ret get_monitor_task_gl();
}

#[doc(hidden)]
fn get_monitor_task_gl() -> hl::high_level_loop {
    let monitor_loop_chan_ptr =
        rustrt::rust_uv_get_kernel_global_chan_ptr();
    log(debug, #fmt("ENTERING global_loop::get() loop chan: %?",
       monitor_loop_chan_ptr));
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
        type hl_loop_req_ch = comm::chan<hl::high_level_loop>;
        let msg_ch = priv::chan_from_global_ptr::<hl_loop_req_ch>(
            monitor_loop_chan_ptr,
            builder_fn) {|msg_po|
            log(debug, "global monitor task starting");
            priv::weaken_task() {|weak_exit_po|
                log(debug, "global monitor task is now weak");
                let hl_loop_data = spawn_high_level_loop();
                let hl_loop = alt hl_loop_data {
                  (async, msg_ch) {
                    hl::high_level_loop({async_handle:async, op_chan:msg_ch})
                  }
                };
                loop {
                    log(debug, "in outer_loop...");
                    let continue = either::either(
                        {|weak_exit|
                            // all normal tasks have ended, tell the
                            // libuv loop to tear_down, then exit
                            log(debug, #fmt("weak_exit_po recv'd msg: %?",
                                           weak_exit));
                            let ( a, loop_msg_ch )= hl_loop_data;
                            comm::send(loop_msg_ch, hl::teardown_loop);
                            ll::async_send(a);
                            false
                        }, {|fetch_ch|
                            log(debug, #fmt("hl_loop req recv'd: %?",
                                           fetch_ch));
                            comm::send(fetch_ch, copy(hl_loop));
                            true
                        }, comm::select2(weak_exit_po, msg_po));
                    if !continue { break; }
                }
                log(debug, "global monitor task is leaving weakend state");
            };
            log(debug, "global monitor task exiting");
        };
        // once we have a chan to the monitor loop, we ask it for
        // the libuv loop's async handle
        let fetch_po = comm::port::<hl::high_level_loop>();
        let fetch_ch = comm::chan(fetch_po);
        comm::send(msg_ch, fetch_ch);
        comm::recv(fetch_po)
    }
}

unsafe fn spawn_high_level_loop() -> (*ll::uv_async_t,
                                      comm::chan<hl::high_level_msg>){
    let exit_po = comm::port::<(*ll::uv_async_t,
                              comm::chan<hl::high_level_msg>)>();
    let exit_ch = comm::chan(exit_po);

    task::spawn_sched(task::manual_threads(1u)) {||
        log(debug, "entering global libuv task");
        let loop_ptr = ll::loop_new();
        priv::weaken_task() {|weak_exit_po|
            log(debug, #fmt("global libuv task is now weak %?",
                            weak_exit_po));
            let loop_msg_po = comm::port::<hl::high_level_msg>();
            let loop_msg_ch = comm::chan(loop_msg_po);
            hl::run_high_level_loop(
                loop_ptr,
                loop_msg_po,
                // before_run
                {|async_handle|
                    log(debug,#fmt("global libuv: before_run %?",
                                  async_handle));
                    let out_data = (async_handle, loop_msg_ch);
                    comm::send(exit_ch, out_data);
                },
                // before_msg_process
                {|async_handle, loop_active|
                    log(debug,#fmt("global libuv: before_msg_drain %? %?",
                                  async_handle, loop_active));
                    true
                },
                // before_tear_down
                {|async_handle|
                    log(debug,#fmt("libuv task: before_tear_down %?",
                                  async_handle));
                }
            );
            log(debug, "global libuv task is leaving weakened state");
        };
        ll::loop_delete(loop_ptr);
        log(debug, "global libuv task exiting");
    };

    comm::recv(exit_po)
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
                             _status: libc::c_int) unsafe {
        log(debug, "in simple timer cb");
        ll::timer_stop(timer_ptr);
        let hl_loop = get_gl();
        hl::interact(hl_loop) {|_loop_ptr|
            log(debug, "closing timer");
            ll::close(timer_ptr, simple_timer_close_cb);
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
    fn test_gl_uv_global_loop_high_level_global_timer() unsafe {
        let hl_loop = get_gl();
        let exit_po = comm::port::<()>();
        let exit_ch = comm::chan(exit_po);
        task::spawn_sched(task::manual_threads(1u), {||
            impl_uv_hl_simple_timer(hl_loop);
            comm::send(exit_ch, ());
        });
        impl_uv_hl_simple_timer(hl_loop);
        comm::recv(exit_po);
    }

    // keeping this test ignored until some kind of stress-test-harness
    // is set up for the build bots
    #[test]
    #[ignore]
    fn test_stress_gl_uv_global_loop_high_level_global_timer() unsafe {
        let hl_loop = get_gl();
        let exit_po = comm::port::<()>();
        let exit_ch = comm::chan(exit_po);
        let cycles = 5000u;
        iter::repeat(cycles) {||
            task::spawn_sched(task::manual_threads(1u), {||
                impl_uv_hl_simple_timer(hl_loop);
                comm::send(exit_ch, ());
            });
        };
        iter::repeat(cycles) {||
            comm::recv(exit_po);
        };
        log(debug, "test_stress_gl_uv_global_loop_high_level_global_timer"+
            " exiting sucessfully!");
    }
}