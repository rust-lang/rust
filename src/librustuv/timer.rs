// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;
use std::rt::rtio::{RtioTimer, Callback};
use std::rt::task::BlockedTask;

use homing::{HomeHandle, HomingIO};
use super::{UvHandle, ForbidUnwind, ForbidSwitch, wait_until_woken_after, Loop};
use uvio::UvIoFactory;
use uvll;

pub struct TimerWatcher {
    pub handle: *uvll::uv_timer_t,
    home: HomeHandle,
    action: Option<NextAction>,
    blocker: Option<BlockedTask>,
    id: uint, // see comments in timer_cb
}

pub enum NextAction {
    WakeTask,
    CallOnce(Box<Callback + Send>),
    CallMany(Box<Callback + Send>, uint),
}

impl TimerWatcher {
    pub fn new(io: &mut UvIoFactory) -> Box<TimerWatcher> {
        let handle = io.make_handle();
        let me = box TimerWatcher::new_home(&io.loop_, handle);
        me.install()
    }

    pub fn new_home(loop_: &Loop, home: HomeHandle) -> TimerWatcher {
        let handle = UvHandle::alloc(None::<TimerWatcher>, uvll::UV_TIMER);
        assert_eq!(unsafe { uvll::uv_timer_init(loop_.handle, handle) }, 0);
        TimerWatcher {
            handle: handle,
            action: None,
            blocker: None,
            home: home,
            id: 0,
        }
    }

    pub fn start(&mut self, f: uvll::uv_timer_cb, msecs: u64, period: u64) {
        assert_eq!(unsafe {
            uvll::uv_timer_start(self.handle, f, msecs, period)
        }, 0)
    }

    pub fn stop(&mut self) {
        assert_eq!(unsafe { uvll::uv_timer_stop(self.handle) }, 0)
    }

    pub unsafe fn set_data<T>(&mut self, data: *T) {
        uvll::set_data_for_uv_handle(self.handle, data);
    }
}

impl HomingIO for TimerWatcher {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle { &mut self.home }
}

impl UvHandle<uvll::uv_timer_t> for TimerWatcher {
    fn uv_handle(&self) -> *uvll::uv_timer_t { self.handle }
}

impl RtioTimer for TimerWatcher {
    fn sleep(&mut self, msecs: u64) {
        // As with all of the below functions, we must be extra careful when
        // destroying the previous action. If the previous action was a channel,
        // destroying it could invoke a context switch. For these situtations,
        // we must temporarily un-home ourselves, then destroy the action, and
        // then re-home again.
        let missile = self.fire_homing_missile();
        self.id += 1;
        self.stop();
        let _missile = match mem::replace(&mut self.action, None) {
            None => missile, // no need to do a homing dance
            Some(action) => {
                drop(missile);      // un-home ourself
                drop(action);       // destroy the previous action
                self.fire_homing_missile()  // re-home ourself
            }
        };

        // If the descheduling operation unwinds after the timer has been
        // started, then we need to call stop on the timer.
        let _f = ForbidUnwind::new("timer");

        self.action = Some(WakeTask);
        wait_until_woken_after(&mut self.blocker, &self.uv_loop(), || {
            self.start(timer_cb, msecs, 0);
        });
        self.stop();
    }

    fn oneshot(&mut self, msecs: u64, cb: Box<Callback + Send>) {
        // similarly to the destructor, we must drop the previous action outside
        // of the homing missile
        let _prev_action = {
            let _m = self.fire_homing_missile();
            self.id += 1;
            self.stop();
            self.start(timer_cb, msecs, 0);
            mem::replace(&mut self.action, Some(CallOnce(cb)))
        };
    }

    fn period(&mut self, msecs: u64, cb: Box<Callback + Send>) {
        // similarly to the destructor, we must drop the previous action outside
        // of the homing missile
        let _prev_action = {
            let _m = self.fire_homing_missile();
            self.id += 1;
            self.stop();
            self.start(timer_cb, msecs, msecs);
            mem::replace(&mut self.action, Some(CallMany(cb, self.id)))
        };
    }
}

extern fn timer_cb(handle: *uvll::uv_timer_t) {
    let _f = ForbidSwitch::new("timer callback can't switch");
    let timer: &mut TimerWatcher = unsafe { UvHandle::from_uv_handle(&handle) };

    match timer.action.take_unwrap() {
        WakeTask => {
            let task = timer.blocker.take_unwrap();
            let _ = task.wake().map(|t| t.reawaken());
        }
        CallOnce(mut cb) => { cb.call() }
        CallMany(mut cb, id) => {
            cb.call();

            // Note that the above operation could have performed some form of
            // scheduling. This means that the timer may have decided to insert
            // some other action to happen. This 'id' keeps track of the updates
            // to the timer, so we only reset the action back to sending on this
            // channel if the id has remained the same. This is essentially a
            // bug in that we have mutably aliasable memory, but that's libuv
            // for you. We're guaranteed to all be running on the same thread,
            // so there's no need for any synchronization here.
            if timer.id == id {
                timer.action = Some(CallMany(cb, id));
            }
        }
    }
}

impl Drop for TimerWatcher {
    fn drop(&mut self) {
        // note that this drop is a little subtle. Dropping a channel which is
        // held internally may invoke some scheduling operations. We can't take
        // the channel unless we're on the home scheduler, but once we're on the
        // home scheduler we should never move. Hence, we take the timer's
        // action item and then move it outside of the homing block.
        let _action = {
            let _m = self.fire_homing_missile();
            self.stop();
            self.close();
            self.action.take()
        };
    }
}
