#[doc ="
Utilities that leverage libuv's `uv_timer_*` API
"];

import uv = uv;
export delayed_send, sleep;

#[doc = "
Wait for timeout period then send provided value over a channel

This call returns immediately. Useful as the building block for a number
of higher-level timer functions.

Is not guaranteed to wait for exactly the specified time, but will wait
for *at least* that period of time.

# Arguments

msecs - a timeout period, in milliseconds, to wait
ch - a channel of type T to send a `val` on
val - a value of type T to send over the provided `ch`
"]
fn delayed_send<T: send>(msecs: uint, ch: comm::chan<T>, val: T) {
    task::spawn() {||
        unsafe {
            let timer_done_po = comm::port::<()>();
            let timer_done_ch = comm::chan(timer_done_po);
            let timer_done_ch_ptr = ptr::addr_of(timer_done_ch);
            let timer = uv::ll::timer_t();
            let timer_ptr = ptr::addr_of(timer);
            let hl_loop = uv::global_loop::get();
            uv::hl::interact(hl_loop) {|loop_ptr|
                uv::hl::ref(hl_loop, timer_ptr);
                let init_result = uv::ll::timer_init(loop_ptr, timer_ptr);
                if (init_result == 0i32) {
                    let start_result = uv::ll::timer_start(
                        timer_ptr, delayed_send_cb, msecs, 0u);
                    if (start_result == 0i32) {
                        uv::ll::set_data_for_uv_handle(
                            timer_ptr,
                            timer_done_ch_ptr as *libc::c_void);
                    }
                    else {
                        let error_msg = uv::ll::get_last_err_info(loop_ptr);
                        fail "timer::delayed_send() start failed: "+error_msg;
                    }
                }
                else {
                    let error_msg = uv::ll::get_last_err_info(loop_ptr);
                    fail "timer::delayed_send() init failed: "+error_msg;
                }
            };
            // delayed_send_cb has been processed by libuv
            comm::recv(timer_done_po);
            // notify the caller immediately
            comm::send(ch, copy(val));
            // then clean up our handle
            uv::hl::unref_and_close(hl_loop, timer_ptr,
                                    delayed_send_close_cb);
            // uv_close for this timer has been processed
            comm::recv(timer_done_po);
        }
    };
}

#[doc = "
Blocks the current task for (at least) the specified time period.

Is not guaranteed to sleep for exactly the specified time, but will sleep
for *at least* that period of time.

# Arguments

* msecs - an amount of time, in milliseconds, for the current task to block
"]
fn sleep(msecs: uint) {
    let exit_po = comm::port::<()>();
    let exit_ch = comm::chan(exit_po);
    delayed_send(msecs, exit_ch, ());
    comm::recv(exit_po);
}

// INTERNAL API
crust fn delayed_send_cb(handle: *uv::ll::uv_timer_t,
                                status: libc::c_int) unsafe {
    log(debug, #fmt("delayed_send_cb handle %? status %?", handle, status));
    let timer_done_ch =
        *(uv::ll::get_data_for_uv_handle(handle) as *comm::chan<()>);
    let stop_result = uv::ll::timer_stop(handle);
    if (stop_result == 0i32) {
        comm::send(timer_done_ch, ());
    }
    else {
        let loop_ptr = uv::ll::get_loop_for_uv_handle(handle);
        let error_msg = uv::ll::get_last_err_info(loop_ptr);
        fail "timer::sleep() init failed: "+error_msg;
    }
}

crust fn delayed_send_close_cb(handle: *uv::ll::uv_timer_t) unsafe {
    log(debug, #fmt("delayed_send_close_cb handle %?", handle));
    let timer_done_ch =
        *(uv::ll::get_data_for_uv_handle(handle) as *comm::chan<()>);
    comm::send(timer_done_ch, ());
}

#[cfg(test)]
mod test {
    #[test]
    fn test_timer_simple_sleep_test() {
        sleep(2000u);
    }
}
