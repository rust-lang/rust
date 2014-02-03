// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc::c_int;
use std::rt::local::Local;
use std::rt::rtio::RtioTimer;
use std::rt::task::{BlockedTask, Task};
use std::util;

use homing::{HomeHandle, HomingIO};
use super::{UvHandle, ForbidUnwind, ForbidSwitch};
use uvio::UvIoFactory;
use uvll;

pub struct TimerWatcher {
    handle: *uvll::uv_timer_t,
    home: HomeHandle,
    action: Option<NextAction>,
    id: uint, // see comments in timer_cb
}

pub enum NextAction {
    WakeTask(BlockedTask),
    SendOnce(Chan<()>),
    SendMany(Chan<()>, uint),
}

impl TimerWatcher {
    pub fn new(io: &mut UvIoFactory) -> ~TimerWatcher {
        let handle = UvHandle::alloc(None::<TimerWatcher>, uvll::UV_TIMER);
        assert_eq!(unsafe {
            uvll::uv_timer_init(io.uv_loop(), handle)
        }, 0);
        let me = ~TimerWatcher {
            handle: handle,
            action: None,
            home: io.make_handle(),
            id: 0,
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
        let _missile = match util::replace(&mut self.action, None) {
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

        let task: ~Task = Local::take();
        task.deschedule(1, |task| {
            self.action = Some(WakeTask(task));
            self.start(msecs, 0);
            Ok(())
        });
        self.stop();
    }

    fn oneshot(&mut self, msecs: u64) -> Port<()> {
        let (port, chan) = Chan::new();

        // similarly to the destructor, we must drop the previous action outside
        // of the homing missile
        let _prev_action = {
            let _m = self.fire_homing_missile();
            self.id += 1;
            self.stop();
            self.start(msecs, 0);
            util::replace(&mut self.action, Some(SendOnce(chan)))
        };

        return port;
    }

    fn period(&mut self, msecs: u64) -> Port<()> {
        let (port, chan) = Chan::new();

        // similarly to the destructor, we must drop the previous action outside
        // of the homing missile
        let _prev_action = {
            let _m = self.fire_homing_missile();
            self.id += 1;
            self.stop();
            self.start(msecs, msecs);
            util::replace(&mut self.action, Some(SendMany(chan, self.id)))
        };

        return port;
    }
}

extern fn timer_cb(handle: *uvll::uv_timer_t, status: c_int) {
    let _f = ForbidSwitch::new("timer callback can't switch");
    assert_eq!(status, 0);
    let timer: &mut TimerWatcher = unsafe { UvHandle::from_uv_handle(&handle) };

    match timer.action.take_unwrap() {
        WakeTask(task) => {
            let _ = task.wake().map(|t| t.reawaken());
        }
        SendOnce(chan) => { let _ = chan.try_send(()); }
        SendMany(chan, id) => {
            let _ = chan.try_send(());

            // Note that the above operation could have performed some form of
            // scheduling. This means that the timer may have decided to insert
            // some other action to happen. This 'id' keeps track of the updates
            // to the timer, so we only reset the action back to sending on this
            // channel if the id has remained the same. This is essentially a
            // bug in that we have mutably aliasable memory, but that's libuv
            // for you. We're guaranteed to all be running on the same thread,
            // so there's no need for any synchronization here.
            if timer.id == id {
                timer.action = Some(SendMany(chan, id));
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

#[cfg(test)]
mod test {
    use std::rt::rtio::RtioTimer;
    use super::super::local_loop;
    use super::TimerWatcher;

    #[test]
    fn oneshot() {
        let mut timer = TimerWatcher::new(local_loop());
        let port = timer.oneshot(1);
        port.recv();
        let port = timer.oneshot(1);
        port.recv();
    }

    #[test]
    fn override() {
        let mut timer = TimerWatcher::new(local_loop());
        let oport = timer.oneshot(1);
        let pport = timer.period(1);
        timer.sleep(1);
        assert_eq!(oport.recv_opt(), None);
        assert_eq!(pport.recv_opt(), None);
        timer.oneshot(1).recv();
    }

    #[test]
    fn period() {
        let mut timer = TimerWatcher::new(local_loop());
        let port = timer.period(1);
        port.recv();
        port.recv();
        let port2 = timer.period(1);
        port2.recv();
        port2.recv();
    }

    #[test]
    fn sleep() {
        let mut timer = TimerWatcher::new(local_loop());
        timer.sleep(1);
        timer.sleep(1);
    }

    #[test] #[should_fail]
    fn oneshot_fail() {
        let mut timer = TimerWatcher::new(local_loop());
        let _port = timer.oneshot(1);
        fail!();
    }

    #[test] #[should_fail]
    fn period_fail() {
        let mut timer = TimerWatcher::new(local_loop());
        let _port = timer.period(1);
        fail!();
    }

    #[test] #[should_fail]
    fn normal_fail() {
        let _timer = TimerWatcher::new(local_loop());
        fail!();
    }

    #[test]
    fn closing_channel_during_drop_doesnt_kill_everything() {
        // see issue #10375
        let mut timer = TimerWatcher::new(local_loop());
        let timer_port = timer.period(1000);

        spawn(proc() {
            let _ = timer_port.recv_opt();
        });

        // when we drop the TimerWatcher we're going to destroy the channel,
        // which must wake up the task on the other end
    }

    #[test]
    fn reset_doesnt_switch_tasks() {
        // similar test to the one above.
        let mut timer = TimerWatcher::new(local_loop());
        let timer_port = timer.period(1000);

        spawn(proc() {
            let _ = timer_port.recv_opt();
        });

        drop(timer.oneshot(1));
    }
    #[test]
    fn reset_doesnt_switch_tasks2() {
        // similar test to the one above.
        let mut timer = TimerWatcher::new(local_loop());
        let timer_port = timer.period(1000);

        spawn(proc() {
            let _ = timer_port.recv_opt();
        });

        timer.sleep(1);
    }

    #[test]
    fn sender_goes_away_oneshot() {
        let port = {
            let mut timer = TimerWatcher::new(local_loop());
            timer.oneshot(1000)
        };
        assert_eq!(port.recv_opt(), None);
    }

    #[test]
    fn sender_goes_away_period() {
        let port = {
            let mut timer = TimerWatcher::new(local_loop());
            timer.period(1000)
        };
        assert_eq!(port.recv_opt(), None);
    }

    #[test]
    fn receiver_goes_away_oneshot() {
        let mut timer1 = TimerWatcher::new(local_loop());
        drop(timer1.oneshot(1));
        let mut timer2 = TimerWatcher::new(local_loop());
        // while sleeping, the prevous timer should fire and not have its
        // callback do something terrible.
        timer2.sleep(2);
    }

    #[test]
    fn receiver_goes_away_period() {
        let mut timer1 = TimerWatcher::new(local_loop());
        drop(timer1.period(1));
        let mut timer2 = TimerWatcher::new(local_loop());
        // while sleeping, the prevous timer should fire and not have its
        // callback do something terrible.
        timer2.sleep(2);
    }
}
