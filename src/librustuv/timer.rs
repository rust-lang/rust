// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::comm::{oneshot, stream, PortOne, ChanOne, SendDeferred};
use std::libc::c_int;
use std::rt::BlockedTask;
use std::rt::local::Local;
use std::rt::rtio::RtioTimer;
use std::rt::sched::{Scheduler, SchedHandle};

use uvll;
use super::{Loop, UvHandle};
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
            uvll::uv_timer_init(loop_.handle, handle)
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

        let _m = self.fire_missiles();
        self.action = Some(SendOnce(chan));
        self.start(msecs, 0);

        return port;
    }

    fn period(&mut self, msecs: u64) -> Port<()> {
        let (port, chan) = stream();

        let _m = self.fire_missiles();
        self.action = Some(SendMany(chan));
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
        SendOnce(chan) => chan.send_deferred(()),
        SendMany(chan) => {
            chan.send_deferred(());
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
    use std::rt::rtio::RtioTimer;
    use super::super::run_uv_loop;

    #[test]
    fn oneshot() {
        do run_uv_loop |l| {
            let mut timer = TimerWatcher::new(l);
            let port = timer.oneshot(1);
            port.recv();
            let port = timer.oneshot(1);
            port.recv();
        }
    }

    #[test]
    fn override() {
        do run_uv_loop |l| {
            let mut timer = TimerWatcher::new(l);
            let oport = timer.oneshot(1);
            let pport = timer.period(1);
            timer.sleep(1);
            assert_eq!(oport.try_recv(), None);
            assert_eq!(pport.try_recv(), None);
            timer.oneshot(1).recv();
        }
    }

    #[test]
    fn period() {
        do run_uv_loop |l| {
            let mut timer = TimerWatcher::new(l);
            let port = timer.period(1);
            port.recv();
            port.recv();
            let port = timer.period(1);
            port.recv();
            port.recv();
        }
    }

    #[test]
    fn sleep() {
        do run_uv_loop |l| {
            let mut timer = TimerWatcher::new(l);
            timer.sleep(1);
            timer.sleep(1);
        }
    }
}
