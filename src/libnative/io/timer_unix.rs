// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Timers for non-linux/non-windows OSes
//!
//! This module implements timers with a worker thread, select(), and a lot of
//! witchcraft that turns out to be horribly inaccurate timers. The unfortunate
//! part is that I'm at a loss of what else to do one these OSes. This is also
//! why linux has a specialized timerfd implementation and windows has its own
//! implementation (they're more accurate than this one).
//!
//! The basic idea is that there is a worker thread that's communicated to via a
//! channel and a pipe, the pipe is used by the worker thread in a select()
//! syscall with a timeout. The timeout is the "next timer timeout" while the
//! channel is used to send data over to the worker thread.
//!
//! Whenever the call to select() times out, then a channel receives a message.
//! Whenever the call returns that the file descriptor has information, then the
//! channel from timers is drained, enqueuing all incoming requests.
//!
//! The actual implementation of the helper thread is a sorted array of
//! timers in terms of target firing date. The target is the absolute time at
//! which the timer should fire. Timers are then re-enqueued after a firing if
//! the repeat boolean is set.
//!
//! Naturally, all this logic of adding times and keeping track of
//! relative/absolute time is a little lossy and not quite exact. I've done the
//! best I could to reduce the amount of calls to 'now()', but there's likely
//! still inaccuracies trickling in here and there.
//!
//! One of the tricky parts of this implementation is that whenever a timer is
//! acted upon, it must cancel whatever the previous action was (if one is
//! active) in order to act like the other implementations of this timer. In
//! order to do this, the timer's inner pointer is transferred to the worker
//! thread. Whenever the timer is modified, it first takes ownership back from
//! the worker thread in order to modify the same data structure. This has the
//! side effect of "cancelling" the previous requests while allowing a
//! re-enqueuing later on.
//!
//! Note that all time units in this file are in *milliseconds*.

use libc;
use std::mem;
use std::os;
use std::ptr;
use std::rt::rtio;
use std::rt::rtio::IoResult;
use std::sync::atomic;
use std::comm;

use io::c;
use io::file::FileDesc;
use io::helper_thread::Helper;

helper_init!(static mut HELPER: Helper<Req>)

pub struct Timer {
    id: uint,
    inner: Option<Box<Inner>>,
}

struct Inner {
    cb: Option<Box<rtio::Callback + Send>>,
    interval: u64,
    repeat: bool,
    target: u64,
    id: uint,
}

#[allow(visible_private_types)]
pub enum Req {
    // Add a new timer to the helper thread.
    NewTimer(Box<Inner>),

    // Remove a timer based on its id and then send it back on the channel
    // provided
    RemoveTimer(uint, Sender<Box<Inner>>),
}

// returns the current time (in milliseconds)
pub fn now() -> u64 {
    unsafe {
        let mut now: libc::timeval = mem::zeroed();
        assert_eq!(c::gettimeofday(&mut now, ptr::mut_null()), 0);
        return (now.tv_sec as u64) * 1000 + (now.tv_usec as u64) / 1000;
    }
}

fn helper(input: libc::c_int, messages: Receiver<Req>, _: ()) {
    let mut set: c::fd_set = unsafe { mem::zeroed() };

    let mut fd = FileDesc::new(input, true);
    let mut timeout: libc::timeval = unsafe { mem::zeroed() };

    // active timers are those which are able to be selected upon (and it's a
    // sorted list, and dead timers are those which have expired, but ownership
    // hasn't yet been transferred back to the timer itself.
    let mut active: Vec<Box<Inner>> = vec![];
    let mut dead = vec![];

    // inserts a timer into an array of timers (sorted by firing time)
    fn insert(t: Box<Inner>, active: &mut Vec<Box<Inner>>) {
        match active.iter().position(|tm| tm.target > t.target) {
            Some(pos) => { active.insert(pos, t); }
            None => { active.push(t); }
        }
    }

    // signals the first requests in the queue, possible re-enqueueing it.
    fn signal(active: &mut Vec<Box<Inner>>,
              dead: &mut Vec<(uint, Box<Inner>)>) {
        let mut timer = match active.shift() {
            Some(timer) => timer, None => return
        };
        let mut cb = timer.cb.take().unwrap();
        cb.call();
        if timer.repeat {
            timer.cb = Some(cb);
            timer.target += timer.interval;
            insert(timer, active);
        } else {
            dead.push((timer.id, timer));
        }
    }

    'outer: loop {
        let timeout = if active.len() == 0 {
            // Empty array? no timeout (wait forever for the next request)
            ptr::mut_null()
        } else {
            let now = now();
            // If this request has already expired, then signal it and go
            // through another iteration
            if active.get(0).target <= now {
                signal(&mut active, &mut dead);
                continue;
            }

            // The actual timeout listed in the requests array is an
            // absolute date, so here we translate the absolute time to a
            // relative time.
            let tm = active.get(0).target - now;
            timeout.tv_sec = (tm / 1000) as libc::time_t;
            timeout.tv_usec = ((tm % 1000) * 1000) as libc::suseconds_t;
            &mut timeout as *mut libc::timeval
        };

        c::fd_set(&mut set, input);
        match unsafe {
            c::select(input + 1, &mut set, ptr::mut_null(),
                      ptr::mut_null(), timeout)
        } {
            // timed out
            0 => signal(&mut active, &mut dead),

            // file descriptor write woke us up, we've got some new requests
            1 => {
                loop {
                    match messages.try_recv() {
                        Err(comm::Disconnected) => {
                            assert!(active.len() == 0);
                            break 'outer;
                        }

                        Ok(NewTimer(timer)) => insert(timer, &mut active),

                        Ok(RemoveTimer(id, ack)) => {
                            match dead.iter().position(|&(i, _)| id == i) {
                                Some(i) => {
                                    let (_, i) = dead.remove(i).unwrap();
                                    ack.send(i);
                                    continue
                                }
                                None => {}
                            }
                            let i = active.iter().position(|i| i.id == id);
                            let i = i.expect("no timer found");
                            let t = active.remove(i).unwrap();
                            ack.send(t);
                        }
                        Err(..) => break
                    }
                }

                // drain the file descriptor
                let mut buf = [0];
                assert_eq!(fd.inner_read(buf).ok().unwrap(), 1);
            }

            -1 if os::errno() == libc::EINTR as int => {}
            n => fail!("helper thread failed in select() with error: {} ({})",
                       n, os::last_os_error())
        }
    }
}

impl Timer {
    pub fn new() -> IoResult<Timer> {
        // See notes above regarding using int return value
        // instead of ()
        unsafe { HELPER.boot(|| {}, helper); }

        static mut ID: atomic::AtomicUint = atomic::INIT_ATOMIC_UINT;
        let id = unsafe { ID.fetch_add(1, atomic::Relaxed) };
        Ok(Timer {
            id: id,
            inner: Some(box Inner {
                cb: None,
                interval: 0,
                target: 0,
                repeat: false,
                id: id,
            })
        })
    }

    pub fn sleep(ms: u64) {
        let mut to_sleep = libc::timespec {
            tv_sec: (ms / 1000) as libc::time_t,
            tv_nsec: ((ms % 1000) * 1000000) as libc::c_long,
        };
        while unsafe { libc::nanosleep(&to_sleep, &mut to_sleep) } != 0 {
            if os::errno() as int != libc::EINTR as int {
                fail!("failed to sleep, but not because of EINTR?");
            }
        }
    }

    fn inner(&mut self) -> Box<Inner> {
        match self.inner.take() {
            Some(i) => i,
            None => {
                let (tx, rx) = channel();
                unsafe { HELPER.send(RemoveTimer(self.id, tx)); }
                rx.recv()
            }
        }
    }
}

impl rtio::RtioTimer for Timer {
    fn sleep(&mut self, msecs: u64) {
        let mut inner = self.inner();
        inner.cb = None; // cancel any previous request
        self.inner = Some(inner);

        Timer::sleep(msecs);
    }

    fn oneshot(&mut self, msecs: u64, cb: Box<rtio::Callback + Send>) {
        let now = now();
        let mut inner = self.inner();

        inner.repeat = false;
        inner.cb = Some(cb);
        inner.interval = msecs;
        inner.target = now + msecs;

        unsafe { HELPER.send(NewTimer(inner)); }
    }

    fn period(&mut self, msecs: u64, cb: Box<rtio::Callback + Send>) {
        let now = now();
        let mut inner = self.inner();

        inner.repeat = true;
        inner.cb = Some(cb);
        inner.interval = msecs;
        inner.target = now + msecs;

        unsafe { HELPER.send(NewTimer(inner)); }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.inner = Some(self.inner());
    }
}
