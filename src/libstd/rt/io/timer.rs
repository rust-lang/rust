// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use comm;
use kinds::Send;
use option::{Option, Some, None};
use result::{Ok, Err};
use rt::io::{io_error};
use rt::rtio::{IoFactory, IoFactoryObject,
               RtioTimer, RtioTimerObject};
use rt::local::Local;

pub struct Timer(~RtioTimerObject);

/// Sleep the current task for `msecs` milliseconds.
pub fn sleep(msecs: u64) {
    let mut timer = Timer::new().expect("timer::sleep: could not create a Timer");

    timer.sleep(msecs)
}

impl Timer {

    pub fn new() -> Option<Timer> {
        let timer = unsafe {
            rtdebug!("Timer::init: borrowing io to init timer");
            let io: *mut IoFactoryObject = Local::unsafe_borrow();
            rtdebug!("about to init timer");
            (*io).timer_init()
        };
        match timer {
            Ok(t) => Some(Timer(t)),
            Err(ioerr) => {
                rtdebug!("Timer::init: failed to init: %?", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn sleep(&mut self, msecs: u64) {
        (**self).sleep(msecs);
    }
}

trait TimedPort<T: Send> {
  fn recv_timeout(self, msecs: u64) -> Option<T>;
}

impl<T: Send> TimedPort<T> for comm::PortOne<T> {

    fn recv_timeout(self, msecs: u64) -> Option<T> {
        let mut tout = msecs;
        let mut timer = Timer::new().unwrap();

        while tout > 0 {
            if self.peek() { return Some(self.recv()); }
            timer.sleep(1000);
            tout -= 1000;
        }

        None
    }
}


impl<T: Send> TimedPort<T> for comm::Port<T> {

    fn recv_timeout(self, msecs: u64) -> Option<T> {
        let mut tout = msecs;
        let mut timer = Timer::new().unwrap();

        while tout > 0 {
            if self.peek() { return Some(self.recv()); }
            timer.sleep(1000);
            tout -= 1000;
        }

        None
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rt::test::*;
    use task;
    use comm;

    #[test]
    fn test_io_timer_sleep_simple() {
        do run_in_mt_newsched_task {
            let timer = Timer::new();
            do timer.map_move |mut t| { t.sleep(1) };
        }
    }

    #[test]
    fn test_io_timer_sleep_standalone() {
        do run_in_mt_newsched_task {
            sleep(1)
        }
    }

    #[test]
    fn test_recv_timeout() {
        do run_in_newsched_task {
            let (p, c) = comm::stream::<int>();
            do task::spawn {
                let mut t = Timer::new().unwrap();
                t.sleep(1000);
                c.send(1);
            }

            assert!(p.recv_timeout(2000).unwrap() == 1);
        }
    }

    #[test]
    fn test_recv_timeout_expire() {
        do run_in_newsched_task {
            let (p, c) = comm::stream::<int>();
            do task::spawn {
                let mut t = Timer::new().unwrap();
                t.sleep(3000);
                c.send(1);
            }

            assert!(p.recv_timeout(1000).is_none());
        }
    }
}
