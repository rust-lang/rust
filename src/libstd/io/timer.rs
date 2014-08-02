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
use io::{IoResult, IoError};
use kinds::Send;
use boxed::Box;
use rt::rtio::{IoFactory, LocalIo, RtioTimer, Callback};

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
    obj: Box<RtioTimer + Send>,
}

struct TimerCallback { tx: Sender<()> }

/// Sleep the current task for the specified duration.
///
/// When provided a zero or negative `duration`, the function will
/// return immediately.
pub fn sleep(duration: Duration) {
    let timer = Timer::new();
    let mut timer = timer.ok().expect("timer::sleep: could not create a Timer");

    timer.sleep(duration)
}

impl Timer {
    /// Creates a new timer which can be used to put the current task to sleep
    /// for a number of milliseconds, or to possibly create channels which will
    /// get notified after an amount of time has passed.
    pub fn new() -> IoResult<Timer> {
        LocalIo::maybe_raise(|io| {
            io.timer_init().map(|t| Timer { obj: t })
        }).map_err(IoError::from_rtio_error)
    }

    /// Blocks the current task for the specified duration.
    ///
    /// Note that this function will cause any other receivers for this timer to
    /// be invalidated (the other end will be closed).
    ///
    /// When provided a zero or negative `duration`, the function will
    /// return immediately.
    pub fn sleep(&mut self, duration: Duration) {
        // Short-circuit the timer backend for 0 duration
        let ms = in_ms_u64(duration);
        if ms == 0 { return }
        self.obj.sleep(ms);
    }

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
    pub fn oneshot(&mut self, duration: Duration) -> Receiver<()> {
        let (tx, rx) = channel();
        // Short-circuit the timer backend for 0 duration
        if in_ms_u64(duration) != 0 {
            self.obj.oneshot(in_ms_u64(duration), box TimerCallback { tx: tx });
        } else {
            tx.send(());
        }
        return rx
    }

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
    pub fn periodic(&mut self, duration: Duration) -> Receiver<()> {
        let ms = in_ms_u64(duration);
        // FIXME: The backend implementations don't ever send a message
        // if given a 0 ms duration. Temporarily using 1ms. It's
        // not clear what use a 0ms period is anyway...
        let ms = if ms == 0 { 1 } else { ms };
        let (tx, rx) = channel();
        self.obj.period(ms, box TimerCallback { tx: tx });
        return rx
    }
}

impl Callback for TimerCallback {
    fn call(&mut self) {
        let _ = self.tx.send_opt(());
    }
}

fn in_ms_u64(d: Duration) -> u64 {
    let ms = d.num_milliseconds();
    if ms < 0 { return 0 };
    return ms as u64;
}

#[cfg(test)]
mod test {
    iotest!(fn test_io_timer_sleep_simple() {
        let mut timer = Timer::new().unwrap();
        timer.sleep(Duration::milliseconds(1));
    })

    iotest!(fn test_io_timer_sleep_oneshot() {
        let mut timer = Timer::new().unwrap();
        timer.oneshot(Duration::milliseconds(1)).recv();
    })

    iotest!(fn test_io_timer_sleep_oneshot_forget() {
        let mut timer = Timer::new().unwrap();
        timer.oneshot(Duration::milliseconds(100000000));
    })

    iotest!(fn oneshot_twice() {
        let mut timer = Timer::new().unwrap();
        let rx1 = timer.oneshot(Duration::milliseconds(10000));
        let rx = timer.oneshot(Duration::milliseconds(1));
        rx.recv();
        assert_eq!(rx1.recv_opt(), Err(()));
    })

    iotest!(fn test_io_timer_oneshot_then_sleep() {
        let mut timer = Timer::new().unwrap();
        let rx = timer.oneshot(Duration::milliseconds(100000000));
        timer.sleep(Duration::milliseconds(1)); // this should invalidate rx

        assert_eq!(rx.recv_opt(), Err(()));
    })

    iotest!(fn test_io_timer_sleep_periodic() {
        let mut timer = Timer::new().unwrap();
        let rx = timer.periodic(Duration::milliseconds(1));
        rx.recv();
        rx.recv();
        rx.recv();
    })

    iotest!(fn test_io_timer_sleep_periodic_forget() {
        let mut timer = Timer::new().unwrap();
        timer.periodic(Duration::milliseconds(100000000));
    })

    iotest!(fn test_io_timer_sleep_standalone() {
        sleep(Duration::milliseconds(1))
    })

    iotest!(fn oneshot() {
        let mut timer = Timer::new().unwrap();

        let rx = timer.oneshot(Duration::milliseconds(1));
        rx.recv();
        assert!(rx.recv_opt().is_err());

        let rx = timer.oneshot(Duration::milliseconds(1));
        rx.recv();
        assert!(rx.recv_opt().is_err());
    })

    iotest!(fn override() {
        let mut timer = Timer::new().unwrap();
        let orx = timer.oneshot(Duration::milliseconds(100));
        let prx = timer.periodic(Duration::milliseconds(100));
        timer.sleep(Duration::milliseconds(1));
        assert_eq!(orx.recv_opt(), Err(()));
        assert_eq!(prx.recv_opt(), Err(()));
        timer.oneshot(Duration::milliseconds(1)).recv();
    })

    iotest!(fn period() {
        let mut timer = Timer::new().unwrap();
        let rx = timer.periodic(Duration::milliseconds(1));
        rx.recv();
        rx.recv();
        let rx2 = timer.periodic(Duration::milliseconds(1));
        rx2.recv();
        rx2.recv();
    })

    iotest!(fn sleep() {
        let mut timer = Timer::new().unwrap();
        timer.sleep(Duration::milliseconds(1));
        timer.sleep(Duration::milliseconds(1));
    })

    iotest!(fn oneshot_fail() {
        let mut timer = Timer::new().unwrap();
        let _rx = timer.oneshot(Duration::milliseconds(1));
        fail!();
    } #[should_fail])

    iotest!(fn period_fail() {
        let mut timer = Timer::new().unwrap();
        let _rx = timer.periodic(Duration::milliseconds(1));
        fail!();
    } #[should_fail])

    iotest!(fn normal_fail() {
        let _timer = Timer::new().unwrap();
        fail!();
    } #[should_fail])

    iotest!(fn closing_channel_during_drop_doesnt_kill_everything() {
        // see issue #10375
        let mut timer = Timer::new().unwrap();
        let timer_rx = timer.periodic(Duration::milliseconds(1000));

        spawn(proc() {
            let _ = timer_rx.recv_opt();
        });

        // when we drop the TimerWatcher we're going to destroy the channel,
        // which must wake up the task on the other end
    })

    iotest!(fn reset_doesnt_switch_tasks() {
        // similar test to the one above.
        let mut timer = Timer::new().unwrap();
        let timer_rx = timer.periodic(Duration::milliseconds(1000));

        spawn(proc() {
            let _ = timer_rx.recv_opt();
        });

        timer.oneshot(Duration::milliseconds(1));
    })

    iotest!(fn reset_doesnt_switch_tasks2() {
        // similar test to the one above.
        let mut timer = Timer::new().unwrap();
        let timer_rx = timer.periodic(Duration::milliseconds(1000));

        spawn(proc() {
            let _ = timer_rx.recv_opt();
        });

        timer.sleep(Duration::milliseconds(1));
    })

    iotest!(fn sender_goes_away_oneshot() {
        let rx = {
            let mut timer = Timer::new().unwrap();
            timer.oneshot(Duration::milliseconds(1000))
        };
        assert_eq!(rx.recv_opt(), Err(()));
    })

    iotest!(fn sender_goes_away_period() {
        let rx = {
            let mut timer = Timer::new().unwrap();
            timer.periodic(Duration::milliseconds(1000))
        };
        assert_eq!(rx.recv_opt(), Err(()));
    })

    iotest!(fn receiver_goes_away_oneshot() {
        let mut timer1 = Timer::new().unwrap();
        timer1.oneshot(Duration::milliseconds(1));
        let mut timer2 = Timer::new().unwrap();
        // while sleeping, the previous timer should fire and not have its
        // callback do something terrible.
        timer2.sleep(Duration::milliseconds(2));
    })

    iotest!(fn receiver_goes_away_period() {
        let mut timer1 = Timer::new().unwrap();
        timer1.periodic(Duration::milliseconds(1));
        let mut timer2 = Timer::new().unwrap();
        // while sleeping, the previous timer should fire and not have its
        // callback do something terrible.
        timer2.sleep(Duration::milliseconds(2));
    })

    iotest!(fn sleep_zero() {
        let mut timer = Timer::new().unwrap();
        timer.sleep(Duration::milliseconds(0));
    })

    iotest!(fn sleep_negative() {
        let mut timer = Timer::new().unwrap();
        timer.sleep(Duration::milliseconds(-1000000));
    })

    iotest!(fn oneshot_zero() {
        let mut timer = Timer::new().unwrap();
        let rx = timer.oneshot(Duration::milliseconds(0));
        rx.recv();
    })

    iotest!(fn oneshot_negative() {
        let mut timer = Timer::new().unwrap();
        let rx = timer.oneshot(Duration::milliseconds(-1000000));
        rx.recv();
    })

    iotest!(fn periodic_zero() {
        let mut timer = Timer::new().unwrap();
        let rx = timer.periodic(Duration::milliseconds(0));
        rx.recv();
        rx.recv();
        rx.recv();
        rx.recv();
    })

    iotest!(fn periodic_negative() {
        let mut timer = Timer::new().unwrap();
        let rx = timer.periodic(Duration::milliseconds(-1000000));
        rx.recv();
        rx.recv();
        rx.recv();
        rx.recv();
    })

}
