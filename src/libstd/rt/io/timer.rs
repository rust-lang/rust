// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    /// for a number of milliseconds.
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

    pub fn sleep(&mut self, msecs: u64) {
        self.obj.sleep(msecs);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rt::test::*;
    #[test]
    fn test_io_timer_sleep_simple() {
        do run_in_mt_newsched_task {
            let timer = Timer::new();
            do timer.map |mut t| { t.sleep(1) };
        }
    }

    #[test]
    fn test_io_timer_sleep_standalone() {
        do run_in_mt_newsched_task {
            sleep(1)
        }
    }
}
