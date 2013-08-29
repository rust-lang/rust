// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{c_void, c_int};
use option::Some;
use rt::uv::uvll;
use rt::uv::{Watcher, Loop, NativeHandle, TimerCallback, NullCallback};
use rt::uv::status_to_maybe_uv_error;

pub struct TimerWatcher(*uvll::uv_timer_t);
impl Watcher for TimerWatcher { }

impl TimerWatcher {
    pub fn new(loop_: &mut Loop) -> TimerWatcher {
        unsafe {
            let handle = uvll::malloc_handle(uvll::UV_TIMER);
            assert!(handle.is_not_null());
            assert!(0 == uvll::timer_init(loop_.native_handle(), handle));
            let mut watcher: TimerWatcher = NativeHandle::from_native_handle(handle);
            watcher.install_watcher_data();
            return watcher;
        }
    }

    pub fn start(&mut self, timeout: u64, repeat: u64, cb: TimerCallback) {
        {
            let data = self.get_watcher_data();
            data.timer_cb = Some(cb);
        }

        unsafe {
            uvll::timer_start(self.native_handle(), timer_cb, timeout, repeat);
        }

        extern fn timer_cb(handle: *uvll::uv_timer_t, status: c_int) {
            let mut watcher: TimerWatcher = NativeHandle::from_native_handle(handle);
            let data = watcher.get_watcher_data();
            let cb = data.timer_cb.get_ref();
            let status = status_to_maybe_uv_error(watcher, status);
            (*cb)(watcher, status);
        }
    }

    pub fn stop(&mut self) {
        unsafe {
            uvll::timer_stop(self.native_handle());
        }
    }

    pub fn close(self, cb: NullCallback) {
        let mut watcher = self;
        {
            let data = watcher.get_watcher_data();
            assert!(data.close_cb.is_none());
            data.close_cb = Some(cb);
        }

        unsafe {
            uvll::close(watcher.native_handle(), close_cb);
        }

        extern fn close_cb(handle: *uvll::uv_timer_t) {
            let mut watcher: TimerWatcher = NativeHandle::from_native_handle(handle);
            {
                let data = watcher.get_watcher_data();
                data.close_cb.take_unwrap()();
            }
            watcher.drop_watcher_data();
            unsafe {
                uvll::free_handle(handle as *c_void);
            }
        }
    }
}

impl NativeHandle<*uvll::uv_timer_t> for TimerWatcher {
    fn from_native_handle(handle: *uvll::uv_timer_t) -> TimerWatcher {
        TimerWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_idle_t {
        match self { &TimerWatcher(ptr) => ptr }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rt::uv::Loop;
    use unstable::run_in_bare_thread;

    #[test]
    fn smoke_test() {
        do run_in_bare_thread {
            let mut count = 0;
            let count_ptr: *mut int = &mut count;
            let mut loop_ = Loop::new();
            let mut timer = TimerWatcher::new(&mut loop_);
            do timer.start(10, 0) |timer, status| {
                assert!(status.is_none());
                unsafe { *count_ptr += 1 };
                timer.close(||());
            }
            loop_.run();
            loop_.close();
            assert!(count == 1);
        }
    }

    #[test]
    fn start_twice() {
        do run_in_bare_thread {
            let mut count = 0;
            let count_ptr: *mut int = &mut count;
            let mut loop_ = Loop::new();
            let mut timer = TimerWatcher::new(&mut loop_);
            do timer.start(10, 0) |timer, status| {
                let mut timer = timer;
                assert!(status.is_none());
                unsafe { *count_ptr += 1 };
                do timer.start(10, 0) |timer, status| {
                    assert!(status.is_none());
                    unsafe { *count_ptr += 1 };
                    timer.close(||());
                }
            }
            loop_.run();
            loop_.close();
            assert!(count == 2);
        }
    }

    #[test]
    fn repeat_stop() {
        do run_in_bare_thread {
            let mut count = 0;
            let count_ptr: *mut int = &mut count;
            let mut loop_ = Loop::new();
            let mut timer = TimerWatcher::new(&mut loop_);
            do timer.start(1, 2) |timer, status| {
                assert!(status.is_none());
                unsafe {
                    *count_ptr += 1;

                    if *count_ptr == 10 {

                        // Stop the timer and do something else
                        let mut timer = timer;
                        timer.stop();
                        // Freeze timer so it can be captured
                        let timer = timer;

                        let mut loop_ = timer.event_loop();
                        let mut timer2 = TimerWatcher::new(&mut loop_);
                        do timer2.start(10, 0) |timer2, _| {

                            *count_ptr += 1;

                            timer2.close(||());

                            // Restart the original timer
                            let mut timer = timer;
                            do timer.start(1, 0) |timer, _| {
                                *count_ptr += 1;
                                timer.close(||());
                            }
                        }
                    }
                };
            }
            loop_.run();
            loop_.close();
            assert!(count == 12);
        }
    }

}
