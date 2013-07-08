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


use uv;
use uv::iotask;
use uv::iotask::IoTask;

use std::cast::transmute;
use std::cast;
use std::comm::{stream, Chan, SharedChan, Port, select2i};
use std::either;
use std::libc::c_void;
use std::libc;

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
pub fn delayed_send<T:Send>(iotask: &IoTask,
                              msecs: uint,
                              ch: &Chan<T>,
                              val: T) {
    let (timer_done_po, timer_done_ch) = stream::<()>();
    let timer_done_ch = SharedChan::new(timer_done_ch);
    let timer = uv::ll::timer_t();
    let timer_ptr: *uv::ll::uv_timer_t = &timer;
    do iotask::interact(iotask) |loop_ptr| {
        unsafe {
            let init_result = uv::ll::timer_init(loop_ptr, timer_ptr);
            if (init_result == 0i32) {
                let start_result = uv::ll::timer_start(
                    timer_ptr, delayed_send_cb, msecs, 0u);
                if (start_result == 0i32) {
                    // Note: putting the channel into a ~
                    // to cast to *c_void
                    let timer_done_ch_clone = ~timer_done_ch.clone();
                    let timer_done_ch_ptr = transmute::<
                        ~SharedChan<()>, *c_void>(
                        timer_done_ch_clone);
                    uv::ll::set_data_for_uv_handle(
                        timer_ptr,
                        timer_done_ch_ptr);
                } else {
                    let error_msg = uv::ll::get_last_err_info(
                        loop_ptr);
                    fail!("timer::delayed_send() start failed: %s", error_msg);
                }
            } else {
                let error_msg = uv::ll::get_last_err_info(loop_ptr);
                fail!("timer::delayed_send() init failed: %s", error_msg);
            }
        }
    };
    // delayed_send_cb has been processed by libuv
    timer_done_po.recv();
    // notify the caller immediately
    ch.send(val);
    // uv_close for this timer has been processed
    timer_done_po.recv();
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
pub fn sleep(iotask: &IoTask, msecs: uint) {
    let (exit_po, exit_ch) = stream::<()>();
    delayed_send(iotask, msecs, &exit_ch, ());
    exit_po.recv();
}

/**
 * Receive on a port for (up to) a specified time, then return an `Option<T>`
 *
 * This call will block to receive on the provided port for up to the
 * specified timeout. Depending on whether the provided port receives in that
 * time period, `recv_timeout` will return an `Option<T>` representing the
 * result.
 *
 * # Arguments
 *
 * * `iotask' - `uv::iotask` that the tcp request will run on
 * * msecs - an mount of time, in milliseconds, to wait to receive
 * * wait_port - a `std::comm::port<T>` to receive on
 *
 * # Returns
 *
 * An `Option<T>` representing the outcome of the call. If the call `recv`'d
 * on the provided port in the allotted timeout period, then the result will
 * be a `Some(T)`. If not, then `None` will be returned.
 */
pub fn recv_timeout<T:Copy + Send>(iotask: &IoTask,
                                   msecs: uint,
                                   wait_po: &Port<T>)
                                   -> Option<T> {
    let (timeout_po, timeout_ch) = stream::<()>();
    let mut timeout_po = timeout_po;
    delayed_send(iotask, msecs, &timeout_ch, ());

    // XXX: Workaround due to ports and channels not being &mut. They should
    // be.
    unsafe {
        let wait_po = cast::transmute_mut(wait_po);

        either::either(
            |_| {
                None
            }, |_| {
                Some(wait_po.recv())
            }, &select2i(&mut timeout_po, wait_po)
        )
    }
}

// INTERNAL API
extern fn delayed_send_cb(handle: *uv::ll::uv_timer_t, status: libc::c_int) {
    unsafe {
        debug!(
            "delayed_send_cb handle %? status %?", handle, status);
        // Faking a borrowed pointer to our ~SharedChan
        let timer_done_ch_ptr: &*c_void = &uv::ll::get_data_for_uv_handle(
            handle);
        let timer_done_ch_ptr = transmute::<&*c_void, &~SharedChan<()>>(
            timer_done_ch_ptr);
        let stop_result = uv::ll::timer_stop(handle);
        if (stop_result == 0i32) {
            timer_done_ch_ptr.send(());
            uv::ll::close(handle, delayed_send_close_cb);
        } else {
            let loop_ptr = uv::ll::get_loop_for_uv_handle(handle);
            let error_msg = uv::ll::get_last_err_info(loop_ptr);
            fail!("timer::sleep() init failed: %s", error_msg);
        }
    }
}

extern fn delayed_send_close_cb(handle: *uv::ll::uv_timer_t) {
    unsafe {
        debug!("delayed_send_close_cb handle %?", handle);
        let timer_done_ch_ptr = uv::ll::get_data_for_uv_handle(handle);
        let timer_done_ch = transmute::<*c_void, ~SharedChan<()>>(
            timer_done_ch_ptr);
        timer_done_ch.send(());
    }
}

#[cfg(test)]
mod test {

    use timer::*;
    use uv;

    use std::cell::Cell;
    use std::pipes::{stream, SharedChan};
    use std::rand::RngUtil;
    use std::rand;
    use std::task;

    #[test]
    fn test_gl_timer_simple_sleep_test() {
        let hl_loop = &uv::global_loop::get();
        sleep(hl_loop, 1u);
    }

    #[test]
    fn test_gl_timer_sleep_stress1() {
        let hl_loop = &uv::global_loop::get();
        for 50u.times {
            sleep(hl_loop, 1u);
        }
    }

    #[test]
    fn test_gl_timer_sleep_stress2() {
        let (po, ch) = stream();
        let ch = SharedChan::new(ch);
        let hl_loop = &uv::global_loop::get();

        let repeat = 20u;
        let spec = {

            ~[(1u,  20u),
             (10u, 10u),
             (20u, 2u)]

        };

        for repeat.times {
            let ch = ch.clone();
            for spec.iter().advance |spec| {
                let (times, maxms) = *spec;
                let ch = ch.clone();
                let hl_loop_clone = hl_loop.clone();
                do task::spawn {
                    use std::rand::*;
                    let mut rng = rng();
                    for times.times {
                        sleep(&hl_loop_clone, rng.next() as uint % maxms);
                    }
                    ch.send(());
                }
            }
        }

        for (repeat * spec.len()).times {
            po.recv()
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

        for (times as uint).times {
            task::yield();

            let expected = rand::rng().gen_str(16u);
            let (test_po, test_ch) = stream::<~str>();

            do task::spawn() {
                delayed_send(hl_loop, 1u, &test_ch, expected);
            };

            match recv_timeout(hl_loop, 10u, &test_po) {
              Some(val) => {
                assert_eq!(val, expected);
                successes += 1;
              }
              _ => failures += 1
            };
        }

        assert!(successes > times / 2);
    }

    #[test]
    fn test_gl_timer_recv_timeout_after_time_passes() {
        let times = 100;
        let mut successes = 0;
        let mut failures = 0;
        let hl_loop = uv::global_loop::get();

        for (times as uint).times {
            let mut rng = rand::rng();
            let expected = Cell::new(rng.gen_str(16u));
            let (test_po, test_ch) = stream::<~str>();
            let hl_loop_clone = hl_loop.clone();
            do task::spawn() {
                delayed_send(&hl_loop_clone, 50u, &test_ch, expected.take());
            };

            match recv_timeout(&hl_loop, 1u, &test_po) {
              None => successes += 1,
              _ => failures += 1
            };
        }

        assert!(successes > times / 2);
    }
}
