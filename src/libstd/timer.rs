// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities that leverage libuv's `uv_timer_*` API

#[forbid(deprecated_mode)];

use uv::iotask;
use uv::iotask::IoTask;

/**
 * Wait for timeout period then send provided value over a channel
 *
 * This call returns immediately. Useful as the building block for a number
 * of higher-level timer functions.
 *
 * Is not guaranteed to wait for exactly the specified time, but will wait
 * for *at least* that period of time.
 *
 * # Arguments
 *
 * * `hl_loop` - a `uv::hl::high_level_loop` that the tcp request will run on
 * * msecs - a timeout period, in milliseconds, to wait
 * * ch - a channel of type T to send a `val` on
 * * val - a value of type T to send over the provided `ch`
 */
pub fn delayed_send<T: Owned>(iotask: IoTask,
                                  msecs: uint, ch: oldcomm::Chan<T>, val: T) {
        unsafe {
            let timer_done_po = oldcomm::Port::<()>();
            let timer_done_ch = oldcomm::Chan(&timer_done_po);
            let timer_done_ch_ptr = ptr::addr_of(&timer_done_ch);
            let timer = uv::ll::timer_t();
            let timer_ptr = ptr::addr_of(&timer);
            do iotask::interact(iotask) |loop_ptr| unsafe {
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
                        fail ~"timer::delayed_send() start failed: " +
                            error_msg;
                    }
                }
                else {
                    let error_msg = uv::ll::get_last_err_info(loop_ptr);
                    fail ~"timer::delayed_send() init failed: "+error_msg;
                }
            };
            // delayed_send_cb has been processed by libuv
            oldcomm::recv(timer_done_po);
            // notify the caller immediately
            oldcomm::send(ch, move(val));
            // uv_close for this timer has been processed
            oldcomm::recv(timer_done_po);
    };
}

/**
 * Blocks the current task for (at least) the specified time period.
 *
 * Is not guaranteed to sleep for exactly the specified time, but will sleep
 * for *at least* that period of time.
 *
 * # Arguments
 *
 * * `iotask` - a `uv::iotask` that the tcp request will run on
 * * msecs - an amount of time, in milliseconds, for the current task to block
 */
pub fn sleep(iotask: IoTask, msecs: uint) {
    let exit_po = oldcomm::Port::<()>();
    let exit_ch = oldcomm::Chan(&exit_po);
    delayed_send(iotask, msecs, exit_ch, ());
    oldcomm::recv(exit_po);
}

/**
 * Receive on a port for (up to) a specified time, then return an `option<T>`
 *
 * This call will block to receive on the provided port for up to the
 * specified timeout. Depending on whether the provided port receives in that
 * time period, `recv_timeout` will return an `option<T>` representing the
 * result.
 *
 * # Arguments
 *
 * * `iotask' - `uv::iotask` that the tcp request will run on
 * * msecs - an mount of time, in milliseconds, to wait to receive
 * * wait_port - a `core::comm::port<T>` to receive on
 *
 * # Returns
 *
 * An `option<T>` representing the outcome of the call. If the call `recv`'d
 * on the provided port in the allotted timeout period, then the result will
 * be a `some(T)`. If not, then `none` will be returned.
 */
pub fn recv_timeout<T: Copy Owned>(iotask: IoTask,
                              msecs: uint,
                              wait_po: oldcomm::Port<T>) -> Option<T> {
    let timeout_po = oldcomm::Port::<()>();
    let timeout_ch = oldcomm::Chan(&timeout_po);
    delayed_send(iotask, msecs, timeout_ch, ());
    // FIXME: This could be written clearer (#2618)
    either::either(
        |left_val| {
            log(debug, fmt!("recv_time .. left_val %?",
                           left_val));
            None
        }, |right_val| {
            Some(*right_val)
        }, &oldcomm::select2(timeout_po, wait_po)
    )
}

// INTERNAL API
extern fn delayed_send_cb(handle: *uv::ll::uv_timer_t,
                                status: libc::c_int) unsafe {
    log(debug, fmt!("delayed_send_cb handle %? status %?", handle, status));
    let timer_done_ch =
        *(uv::ll::get_data_for_uv_handle(handle) as *oldcomm::Chan<()>);
    let stop_result = uv::ll::timer_stop(handle);
    if (stop_result == 0i32) {
        oldcomm::send(timer_done_ch, ());
        uv::ll::close(handle, delayed_send_close_cb);
    }
    else {
        let loop_ptr = uv::ll::get_loop_for_uv_handle(handle);
        let error_msg = uv::ll::get_last_err_info(loop_ptr);
        fail ~"timer::sleep() init failed: "+error_msg;
    }
}

extern fn delayed_send_close_cb(handle: *uv::ll::uv_timer_t) unsafe {
    log(debug, fmt!("delayed_send_close_cb handle %?", handle));
    let timer_done_ch =
        *(uv::ll::get_data_for_uv_handle(handle) as *oldcomm::Chan<()>);
    oldcomm::send(timer_done_ch, ());
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    #[test]
    fn test_gl_timer_simple_sleep_test() {
        let hl_loop = uv::global_loop::get();
        sleep(hl_loop, 1u);
    }

    #[test]
    fn test_gl_timer_sleep_stress1() {
        let hl_loop = uv::global_loop::get();
        for iter::repeat(50u) {
            sleep(hl_loop, 1u);
        }
    }

    #[test]
    fn test_gl_timer_sleep_stress2() {
        let po = oldcomm::Port();
        let ch = oldcomm::Chan(&po);
        let hl_loop = uv::global_loop::get();

        let repeat = 20u;
        let spec = {

            ~[(1u,  20u),
             (10u, 10u),
             (20u, 2u)]

        };

        for iter::repeat(repeat) {

            for spec.each |spec| {
                let (times, maxms) = *spec;
                do task::spawn {
                    use rand::*;
                    let rng = Rng();
                    for iter::repeat(times) {
                        sleep(hl_loop, rng.next() as uint % maxms);
                    }
                    oldcomm::send(ch, ());
                }
            }
        }

        for iter::repeat(repeat * spec.len()) {
            oldcomm::recv(po)
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

        for iter::repeat(times as uint) {
            task::yield();

            let expected = rand::rng().gen_str(16u);
            let test_po = core::comm::port::<str>();
            let test_ch = core::comm::chan(test_po);

            do task::spawn() {
                delayed_send(hl_loop, 1u, test_ch, expected);
            };

            match recv_timeout(hl_loop, 10u, test_po) {
              some(val) => {
                assert val == expected;
                successes += 1;
              }
              _ => failures += 1
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

        for iter::repeat(times as uint) {
            let expected = rand::Rng().gen_str(16u);
            let test_po = oldcomm::Port::<~str>();
            let test_ch = oldcomm::Chan(&test_po);

            do task::spawn() {
                delayed_send(hl_loop, 50u, test_ch, expected);
            };

            match recv_timeout(hl_loop, 1u, test_po) {
              None => successes += 1,
              _ => failures += 1
            };
        }

        assert successes > times / 2;
    }
}
