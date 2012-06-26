#[doc ="
Utilities that leverage libuv's `uv_timer_*` API
"];

import uv = uv;
import uv::iotask;
import iotask::iotask;
export delayed_send, sleep, recv_timeout;

#[doc = "
Wait for timeout period then send provided value over a channel

This call returns immediately. Useful as the building block for a number
of higher-level timer functions.

Is not guaranteed to wait for exactly the specified time, but will wait
for *at least* that period of time.

# Arguments

* `hl_loop` - a `uv::hl::high_level_loop` that the tcp request will run on
* msecs - a timeout period, in milliseconds, to wait
* ch - a channel of type T to send a `val` on
* val - a value of type T to send over the provided `ch`
"]
fn delayed_send<T: copy send>(iotask: iotask,
                              msecs: uint, ch: comm::chan<T>, val: T) {
        unsafe {
            let timer_done_po = comm::port::<()>();
            let timer_done_ch = comm::chan(timer_done_po);
            let timer_done_ch_ptr = ptr::addr_of(timer_done_ch);
            let timer = uv::ll::timer_t();
            let timer_ptr = ptr::addr_of(timer);
            do iotask::interact(iotask) {|loop_ptr|
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
            // uv_close for this timer has been processed
            comm::recv(timer_done_po);
    };
}

#[doc = "
Blocks the current task for (at least) the specified time period.

Is not guaranteed to sleep for exactly the specified time, but will sleep
for *at least* that period of time.

# Arguments

* `iotask` - a `uv::iotask` that the tcp request will run on
* msecs - an amount of time, in milliseconds, for the current task to block
"]
fn sleep(iotask: iotask, msecs: uint) {
    let exit_po = comm::port::<()>();
    let exit_ch = comm::chan(exit_po);
    delayed_send(iotask, msecs, exit_ch, ());
    comm::recv(exit_po);
}

#[doc = "
Receive on a port for (up to) a specified time, then return an `option<T>`

This call will block to receive on the provided port for up to the specified
timeout. Depending on whether the provided port receives in that time period,
`recv_timeout` will return an `option<T>` representing the result.

# Arguments

* `iotask' - `uv::iotask` that the tcp request will run on
* msecs - an mount of time, in milliseconds, to wait to receive
* wait_port - a `comm::port<T>` to receive on

# Returns

An `option<T>` representing the outcome of the call. If the call `recv`'d on
the provided port in the allotted timeout period, then the result will be a
`some(T)`. If not, then `none` will be returned.
"]
fn recv_timeout<T: copy send>(iotask: iotask,
                              msecs: uint,
                              wait_po: comm::port<T>) -> option<T> {
    let timeout_po = comm::port::<()>();
    let timeout_ch = comm::chan(timeout_po);
    delayed_send(iotask, msecs, timeout_ch, ());
    // FIXME: This could be written clearer (#2618)
    either::either(
        {|left_val|
            log(debug, #fmt("recv_time .. left_val %?",
                           left_val));
            none
        }, {|right_val|
            some(right_val)
        }, comm::select2(timeout_po, wait_po)
    )
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
        uv::ll::close(handle, delayed_send_close_cb);
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
    fn test_gl_timer_simple_sleep_test() {
        let hl_loop = uv::global_loop::get();
        sleep(hl_loop, 1u);
    }

    #[test]
    fn test_gl_timer_sleep_stress1() {
        let hl_loop = uv::global_loop::get();
        do iter::repeat(200u) {||
            sleep(hl_loop, 1u);
        }
    }

    #[test]
    fn test_gl_timer_sleep_stress2() {
        let po = comm::port();
        let ch = comm::chan(po);
        let hl_loop = uv::global_loop::get();

        let repeat = 20u;
        let spec = {

            ~[(1u,  20u),
             (10u, 10u),
             (20u, 2u)]

        };

        do iter::repeat(repeat) {||

            for spec.each {|spec|
                let (times, maxms) = spec;
                do task::spawn {||
                    import rand::*;
                    let rng = rng();
                    do iter::repeat(times) {||
                        sleep(hl_loop, rng.next() as uint % maxms);
                    }
                    comm::send(ch, ());
                }
            }
        }

        do iter::repeat(repeat * spec.len()) {||
            comm::recv(po)
        }
    }

    // Because valgrind serializes multithreaded programs it can
    // make timing-sensitive tests fail in wierd ways. In these
    // next test we run them many times and expect them to pass
    // the majority of tries.

    #[test]
    #[cfg(ignore)]
    fn test_gl_timer_recv_timeout_before_time_passes() {
        let times = 100;
        let mut successes = 0;
        let mut failures = 0;
        let hl_loop = uv::global_loop::get();

        do iter::repeat(times as uint) {||
            task::yield();

            let expected = rand::rng().gen_str(16u);
            let test_po = comm::port::<str>();
            let test_ch = comm::chan(test_po);

            do task::spawn() {||
                delayed_send(hl_loop, 1u, test_ch, expected);
            };

            alt recv_timeout(hl_loop, 10u, test_po) {
              some(val) { assert val == expected; successes += 1; }
              _ { failures += 1; }
            };
        }

        assert successes > times / 2;
    }

    #[test]
    fn test_gl_timer_recv_timeout_after_time_passes() {
        let times = 100;
        let mut successes = 0;
        let mut failures = 0;
        let hl_loop = uv::global_loop::get();

        do iter::repeat(times as uint) {||
            let expected = rand::rng().gen_str(16u);
            let test_po = comm::port::<str>();
            let test_ch = comm::chan(test_po);

            do task::spawn() {||
                delayed_send(hl_loop, 1000u, test_ch, expected);
            };

            alt recv_timeout(hl_loop, 1u, test_po) {
              none { successes += 1; }
              _ { failures += 1; }
            };
        }

        assert successes > times / 2;
    }
}
