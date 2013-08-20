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
use rt::io::{io_error};
use rt::rtio::{IoFactory, IoFactoryObject,
               RtioTimer, RtioTimerObject};
use rt::local::Local;

pub struct Timer(~RtioTimerObject);

impl Timer {

    pub fn new() -> Option<Timer> {
        let timer = unsafe {
            rtdebug!("Timer::init: borrowing io to init timer");
            let io = Local::unsafe_borrow::<IoFactoryObject>();
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

#[cfg(test)]
mod test {
    use super::*;
    use rt::test::*;
    #[test]
    fn test_io_timer_sleep_simple() {
        do run_in_newsched_task {
            let timer = Timer::new();
            do timer.map_move |mut t| { t.sleep(1) };
        }
    }
}
