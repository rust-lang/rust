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

```rust,ignore

use std::io::Timer;

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

use comm::Port;
use option::Option;
use rt::rtio::{IoFactory, LocalIo, RtioTimer};

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
        LocalIo::maybe_raise(|io| io.timer_init().map(|t| Timer { obj: t }))
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
    pub fn oneshot(&mut self, msecs: u64) -> Port<()> {
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
    iotest!(fn test_io_timer_sleep_simple() {
        let mut timer = Timer::new().unwrap();
        timer.sleep(1);
    })

    iotest!(fn test_io_timer_sleep_oneshot() {
        let mut timer = Timer::new().unwrap();
        timer.oneshot(1).recv();
    })

    iotest!(fn test_io_timer_sleep_oneshot_forget() {
        let mut timer = Timer::new().unwrap();
        timer.oneshot(100000000000);
    })

    iotest!(fn oneshot_twice() {
        let mut timer = Timer::new().unwrap();
        let port1 = timer.oneshot(10000);
        let port = timer.oneshot(1);
        port.recv();
        assert_eq!(port1.recv_opt(), None);
    })

    iotest!(fn test_io_timer_oneshot_then_sleep() {
        let mut timer = Timer::new().unwrap();
        let port = timer.oneshot(100000000000);
        timer.sleep(1); // this should invalidate the port

        assert_eq!(port.recv_opt(), None);
    })

    iotest!(fn test_io_timer_sleep_periodic() {
        let mut timer = Timer::new().unwrap();
        let port = timer.periodic(1);
        port.recv();
        port.recv();
        port.recv();
    })

    iotest!(fn test_io_timer_sleep_periodic_forget() {
        let mut timer = Timer::new().unwrap();
        timer.periodic(100000000000);
    })

    iotest!(fn test_io_timer_sleep_standalone() {
        sleep(1)
    })

    iotest!(fn oneshot() {
        let mut timer = Timer::new().unwrap();

        let port = timer.oneshot(1);
        port.recv();
        assert!(port.recv_opt().is_none());

        let port = timer.oneshot(1);
        port.recv();
        assert!(port.recv_opt().is_none());
    })

    iotest!(fn override() {
        let mut timer = Timer::new().unwrap();
        let oport = timer.oneshot(100);
        let pport = timer.periodic(100);
        timer.sleep(1);
        assert_eq!(oport.recv_opt(), None);
        assert_eq!(pport.recv_opt(), None);
        timer.oneshot(1).recv();
    })

    iotest!(fn period() {
        let mut timer = Timer::new().unwrap();
        let port = timer.periodic(1);
        port.recv();
        port.recv();
        let port2 = timer.periodic(1);
        port2.recv();
        port2.recv();
    })

    iotest!(fn sleep() {
        let mut timer = Timer::new().unwrap();
        timer.sleep(1);
        timer.sleep(1);
    })

    iotest!(fn oneshot_fail() {
        let mut timer = Timer::new().unwrap();
        let _port = timer.oneshot(1);
        fail!();
    } #[should_fail])

    iotest!(fn period_fail() {
        let mut timer = Timer::new().unwrap();
        let _port = timer.periodic(1);
        fail!();
    } #[should_fail])

    iotest!(fn normal_fail() {
        let _timer = Timer::new().unwrap();
        fail!();
    } #[should_fail])

    iotest!(fn closing_channel_during_drop_doesnt_kill_everything() {
        // see issue #10375
        let mut timer = Timer::new().unwrap();
        let timer_port = timer.periodic(1000);

        do spawn {
            timer_port.recv_opt();
        }

        // when we drop the TimerWatcher we're going to destroy the channel,
        // which must wake up the task on the other end
    })

    iotest!(fn reset_doesnt_switch_tasks() {
        // similar test to the one above.
        let mut timer = Timer::new().unwrap();
        let timer_port = timer.periodic(1000);

        do spawn {
            timer_port.recv_opt();
        }

        timer.oneshot(1);
    })

    iotest!(fn reset_doesnt_switch_tasks2() {
        // similar test to the one above.
        let mut timer = Timer::new().unwrap();
        let timer_port = timer.periodic(1000);

        do spawn {
            timer_port.recv_opt();
        }

        timer.sleep(1);
    })

    iotest!(fn sender_goes_away_oneshot() {
        let port = {
            let mut timer = Timer::new().unwrap();
            timer.oneshot(1000)
        };
        assert_eq!(port.recv_opt(), None);
    })

    iotest!(fn sender_goes_away_period() {
        let port = {
            let mut timer = Timer::new().unwrap();
            timer.periodic(1000)
        };
        assert_eq!(port.recv_opt(), None);
    })

    iotest!(fn receiver_goes_away_oneshot() {
        let mut timer1 = Timer::new().unwrap();
        timer1.oneshot(1);
        let mut timer2 = Timer::new().unwrap();
        // while sleeping, the prevous timer should fire and not have its
        // callback do something terrible.
        timer2.sleep(2);
    })

    iotest!(fn receiver_goes_away_period() {
        let mut timer1 = Timer::new().unwrap();
        timer1.periodic(1);
        let mut timer2 = Timer::new().unwrap();
        // while sleeping, the prevous timer should fire and not have its
        // callback do something terrible.
        timer2.sleep(2);
    })
}
