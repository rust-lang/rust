// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Timers for non-Linux/non-Windows OSes
//!
//! This module implements timers with a worker thread, select(), and a lot of
//! witchcraft that turns out to be horribly inaccurate timers. The unfortunate
//! part is that I'm at a loss of what else to do one these OSes. This is also
//! why Linux has a specialized timerfd implementation and windows has its own
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

pub use self::Req::*;

use libc;
use mem;
use os;
use ptr;
use sync::atomic;
use comm;
use sys::c;
use sys::fs::FileDesc;
use sys_common::helper_thread::Helper;
use prelude::*;
use io::IoResult;

helper_init!(static HELPER: Helper<Req>)

pub trait Callback {
    fn call(&mut self);
}

pub struct Timer {
    id: uint,
    inner: Option<Box<Inner>>,
}

pub struct Inner {
    cb: Option<Box<Callback + Send>>,
    interval: u64,
    repeat: bool,
    target: u64,
    id: uint,
}

pub enum Req {
    // Add a new timer to the helper thread.
    NewTimer(Box<Inner>),

    // Remove a timer based on its id and then send it back on the channel
    // provided
    RemoveTimer(uint, Sender<Box<Inner>>),
}

// returns the current time (in milliseconds)
pub fn now() -> u64 { unimplemented!() }

fn helper(input: libc::c_int, messages: Receiver<Req>, _: ()) { unimplemented!() }

impl Timer {
    pub fn new() -> IoResult<Timer> { unimplemented!() }

    pub fn sleep(&mut self, ms: u64) { unimplemented!() }

    pub fn oneshot(&mut self, msecs: u64, cb: Box<Callback + Send>) { unimplemented!() }

    pub fn period(&mut self, msecs: u64, cb: Box<Callback + Send>) { unimplemented!() }

    fn inner(&mut self) -> Box<Inner> { unimplemented!() }
}

impl Drop for Timer {
    fn drop(&mut self) { unimplemented!() }
}
