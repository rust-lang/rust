// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;
use std::comm::{oneshot, stream, PortOne, ChanOne};
use std::libc::c_int;
use std::rt::BlockedTask;
use std::rt::local::Local;
use std::rt::rtio::RtioTimer;
use std::rt::sched::{Scheduler, SchedHandle};

use uvll;
use super::{Loop, NativeHandle, UvHandle};
use uvio::HomingIO;

pub struct TimerWatcher {
    handle: *uvll::uv_timer_t,
    home: SchedHandle,
    action: Option<NextAction>,
}

pub enum NextAction {
    WakeTask(BlockedTask),
    SendOnce(ChanOne<()>),
    SendMany(Chan<()>),
}

impl TimerWatcher {
    pub fn new(loop_: &mut Loop) -> ~TimerWatcher {
        let handle = UvHandle::alloc(None::<TimerWatcher>, uvll::UV_TIMER);
        assert_eq!(unsafe {
            uvll::uv_timer_init(loop_.native_handle(), handle)
        }, 0);
        let me = ~TimerWatcher {
            handle: handle,
            action: None,
            home: get_handle_to_current_scheduler!(),
        };
        return me.install();
    }

    fn start(&mut self, msecs: u64, period: u64) {
        assert_eq!(unsafe {
            uvll::uv_timer_start(self.handle, timer_cb, msecs, period)
        }, 0)
    }

    fn stop(&mut self) {
        assert_eq!(unsafe { uvll::uv_timer_stop(self.handle) }, 0)
    }
}

impl HomingIO for TimerWatcher {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvHandle<uvll::uv_timer_t> for TimerWatcher {
    fn uv_handle(&self) -> *uvll::uv_timer_t { self.handle }
}

impl RtioTimer for TimerWatcher {
    fn sleep(&mut self, msecs: u64) {
        let (_m, sched) = self.fire_missiles_sched();
        do sched.deschedule_running_task_and_then |_sched, task| {
            self.action = Some(WakeTask(task));
            self.start(msecs, 0);
        }
        self.stop();
    }

    fn oneshot(&mut self, msecs: u64) -> PortOne<()> {
        let (port, chan) = oneshot();
        let chan = Cell::new(chan);

        let _m = self.fire_missiles();
        self.action = Some(SendOnce(chan.take()));
        self.start(msecs, 0);

        return port;
    }

    fn period(&mut self, msecs: u64) -> Port<()> {
        let (port, chan) = stream();
        let chan = Cell::new(chan);

        let _m = self.fire_missiles();
        self.action = Some(SendMany(chan.take()));
        self.start(msecs, msecs);

        return port;
    }
}

extern fn timer_cb(handle: *uvll::uv_timer_t, _status: c_int) {
    let timer: &mut TimerWatcher = unsafe { UvHandle::from_uv_handle(&handle) };

    match timer.action.take_unwrap() {
        WakeTask(task) => {
            let sched: ~Scheduler = Local::take();
            sched.resume_blocked_task_immediately(task);
        }
        SendOnce(chan) => chan.send(()),
        SendMany(chan) => {
            chan.send(());
            timer.action = Some(SendMany(chan));
        }
    }
}

impl Drop for TimerWatcher {
    fn drop(&mut self) {
        let _m = self.fire_missiles();
        self.action = None;
        self.stop();
        self.close_async_();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use Loop;
    use std::unstable::run_in_bare_thread;

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
