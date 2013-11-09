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
and create ports which will receive notifications after a period of time.

# Example

```rust

use std::rt::io::Timer;

let mut timer = Timer::new().unwrap();
timer.sleep(10); // block the task for awhile

let timeout = timer.oneshot(10);
// do some work
timeout.recv(); // wait for the timeout to expire

let periodic = timer.periodic(10);
loop {
    periodic.recv();
    // this loop is only executed once every 10ms
}

```

*/

use comm::{Port, PortOne};
use option::{Option, Some, None};
use result::{Ok, Err};
use rt::io::io_error;
use rt::rtio::{IoFactory, RtioTimer, with_local_io};

pub struct Timer {
    priv obj: ~RtioTimer
}

/// Sleep the current task for `msecs` milliseconds.
pub fn sleep(msecs: u64) {
    let mut timer = Timer::new().expect("timer::sleep: could not create a Timer");

    timer.sleep(msecs)
}

impl Timer {
    /// Creates a new timer which can be used to put the current task to sleep
    /// for a number of milliseconds, or to possibly create channels which will
    /// get notified after an amount of time has passed.
    pub fn new() -> Option<Timer> {
        do with_local_io |io| {
            match io.timer_init() {
                Ok(t) => Some(Timer { obj: t }),
                Err(ioerr) => {
                    rtdebug!("Timer::init: failed to init: {:?}", ioerr);
                    io_error::cond.raise(ioerr);
                    None
                }
            }

        }
    }

    /// Blocks the current task for `msecs` milliseconds.
    ///
    /// Note that this function will cause any other ports for this timer to be
    /// invalidated (the other end will be closed).
    pub fn sleep(&mut self, msecs: u64) {
        self.obj.sleep(msecs);
    }

    /// Creates a oneshot port which will have a notification sent when `msecs`
    /// milliseconds has elapsed. This does *not* block the current task, but
    /// instead returns immediately.
    ///
    /// Note that this invalidates any previous port which has been created by
    /// this timer, and that the returned port will be invalidated once the
    /// timer is destroyed (when it falls out of scope).
    pub fn oneshot(&mut self, msecs: u64) -> PortOne<()> {
        self.obj.oneshot(msecs)
    }

    /// Creates a port which will have a continuous stream of notifications
    /// being sent every `msecs` milliseconds. This does *not* block the
    /// current task, but instead returns immediately. The first notification
    /// will not be received immediately, but rather after `msec` milliseconds
    /// have passed.
    ///
    /// Note that this invalidates any previous port which has been created by
    /// this timer, and that the returned port will be invalidated once the
    /// timer is destroyed (when it falls out of scope).
    pub fn periodic(&mut self, msecs: u64) -> Port<()> {
        self.obj.period(msecs)
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;
    use rt::test::*;
    use cell::Cell;
    use task;

    #[test]
    fn test_io_timer_sleep_simple() {
        do run_in_mt_newsched_task {
            let mut timer = Timer::new().unwrap();
            timer.sleep(1);
        }
    }

    #[test]
    fn test_io_timer_sleep_oneshot() {
        do run_in_mt_newsched_task {
            let mut timer = Timer::new().unwrap();
            timer.oneshot(1).recv();
        }
    }

    #[test]
    fn test_io_timer_sleep_oneshot_forget() {
        do run_in_mt_newsched_task {
            let mut timer = Timer::new().unwrap();
            timer.oneshot(100000000000);
        }
    }

    #[test]
    fn oneshot_twice() {
        do run_in_mt_newsched_task {
            let mut timer = Timer::new().unwrap();
            let port1 = timer.oneshot(10000);
            let port = timer.oneshot(1);
            port.recv();
            assert_eq!(port1.try_recv(), None);
        }
    }

    #[test]
    fn test_io_timer_oneshot_then_sleep() {
        do run_in_mt_newsched_task {
            let mut timer = Timer::new().unwrap();
            let port = timer.oneshot(100000000000);
            timer.sleep(1); // this should invalidate the port

            assert_eq!(port.try_recv(), None);
        }
    }

    #[test]
    fn test_io_timer_sleep_periodic() {
        do run_in_mt_newsched_task {
            let mut timer = Timer::new().unwrap();
            let port = timer.periodic(1);
            port.recv();
            port.recv();
            port.recv();
        }
    }

    #[test]
    fn test_io_timer_sleep_periodic_forget() {
        do run_in_mt_newsched_task {
            let mut timer = Timer::new().unwrap();
            timer.periodic(100000000000);
        }
    }

    #[test]
    fn test_io_timer_sleep_standalone() {
        do run_in_mt_newsched_task {
            sleep(1)
        }
    }
}
