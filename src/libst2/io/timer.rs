// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Synchronous Timers

This module exposes the functionality to create timers, block the current task,
and create receivers which will receive notifications after a period of time.

*/

// FIXME: These functions take Durations but only pass ms to the backend impls.

use comm::{Receiver, Sender, channel};
use time::Duration;
use io::IoResult;
use sys::timer::Callback;
use sys::timer::Timer as TimerImp;

/// A synchronous timer object
///
/// Values of this type can be used to put the current task to sleep for a
/// period of time. Handles to this timer can also be created in the form of
/// receivers which will receive notifications over time.
///
/// # Example
///
/// ```
/// # fn main() {}
/// # fn foo() {
/// use std::io::Timer;
/// use std::time::Duration;
///
/// let mut timer = Timer::new().unwrap();
/// timer.sleep(Duration::milliseconds(10)); // block the task for awhile
///
/// let timeout = timer.oneshot(Duration::milliseconds(10));
/// // do some work
/// timeout.recv(); // wait for the timeout to expire
///
/// let periodic = timer.periodic(Duration::milliseconds(10));
/// loop {
///     periodic.recv();
///     // this loop is only executed once every 10ms
/// }
/// # }
/// ```
///
/// If only sleeping is necessary, then a convenience API is provided through
/// the `io::timer` module.
///
/// ```
/// # fn main() {}
/// # fn foo() {
/// use std::io::timer;
/// use std::time::Duration;
///
/// // Put this task to sleep for 5 seconds
/// timer::sleep(Duration::seconds(5));
/// # }
/// ```
pub struct Timer {
    inner: TimerImp,
}

struct TimerCallback { tx: Sender<()> }

/// Sleep the current task for the specified duration.
///
/// When provided a zero or negative `duration`, the function will
/// return immediately.
pub fn sleep(duration: Duration) { unimplemented!() }

impl Timer {
    /// Creates a new timer which can be used to put the current task to sleep
    /// for a number of milliseconds, or to possibly create channels which will
    /// get notified after an amount of time has passed.
    pub fn new() -> IoResult<Timer> { unimplemented!() }

    /// Blocks the current task for the specified duration.
    ///
    /// Note that this function will cause any other receivers for this timer to
    /// be invalidated (the other end will be closed).
    ///
    /// When provided a zero or negative `duration`, the function will
    /// return immediately.
    pub fn sleep(&mut self, duration: Duration) { unimplemented!() }

    /// Creates a oneshot receiver which will have a notification sent when
    /// the specified duration has elapsed.
    ///
    /// This does *not* block the current task, but instead returns immediately.
    ///
    /// Note that this invalidates any previous receiver which has been created
    /// by this timer, and that the returned receiver will be invalidated once
    /// the timer is destroyed (when it falls out of scope). In particular, if
    /// this is called in method-chaining style, the receiver will be
    /// invalidated at the end of that statement, and all `recv` calls will
    /// fail.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io::Timer;
    /// use std::time::Duration;
    ///
    /// let mut timer = Timer::new().unwrap();
    /// let ten_milliseconds = timer.oneshot(Duration::milliseconds(10));
    ///
    /// for _ in range(0u, 100) { /* do work */ }
    ///
    /// // blocks until 10 ms after the `oneshot` call
    /// ten_milliseconds.recv();
    /// ```
    ///
    /// ```rust
    /// use std::io::Timer;
    /// use std::time::Duration;
    ///
    /// // Incorrect, method chaining-style:
    /// let mut five_ms = Timer::new().unwrap().oneshot(Duration::milliseconds(5));
    /// // The timer object was destroyed, so this will always fail:
    /// // five_ms.recv()
    /// ```
    ///
    /// When provided a zero or negative `duration`, the message will
    /// be sent immediately.
    pub fn oneshot(&mut self, duration: Duration) -> Receiver<()> { unimplemented!() }

    /// Creates a receiver which will have a continuous stream of notifications
    /// being sent each time the specified duration has elapsed.
    ///
    /// This does *not* block the current task, but instead returns
    /// immediately. The first notification will not be received immediately,
    /// but rather after the first duration.
    ///
    /// Note that this invalidates any previous receiver which has been created
    /// by this timer, and that the returned receiver will be invalidated once
    /// the timer is destroyed (when it falls out of scope). In particular, if
    /// this is called in method-chaining style, the receiver will be
    /// invalidated at the end of that statement, and all `recv` calls will
    /// fail.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::io::Timer;
    /// use std::time::Duration;
    ///
    /// let mut timer = Timer::new().unwrap();
    /// let ten_milliseconds = timer.periodic(Duration::milliseconds(10));
    ///
    /// for _ in range(0u, 100) { /* do work */ }
    ///
    /// // blocks until 10 ms after the `periodic` call
    /// ten_milliseconds.recv();
    ///
    /// for _ in range(0u, 100) { /* do work */ }
    ///
    /// // blocks until 20 ms after the `periodic` call (*not* 10ms after the
    /// // previous `recv`)
    /// ten_milliseconds.recv();
    /// ```
    ///
    /// ```rust
    /// use std::io::Timer;
    /// use std::time::Duration;
    ///
    /// // Incorrect, method chaining-style.
    /// let mut five_ms = Timer::new().unwrap().periodic(Duration::milliseconds(5));
    /// // The timer object was destroyed, so this will always fail:
    /// // five_ms.recv()
    /// ```
    ///
    /// When provided a zero or negative `duration`, the messages will
    /// be sent without delay.
    pub fn periodic(&mut self, duration: Duration) -> Receiver<()> { unimplemented!() }
}

impl Callback for TimerCallback {
    fn call(&mut self) { unimplemented!() }
}

fn in_ms_u64(d: Duration) -> u64 { unimplemented!() }
