// copyright 2013 the rust project developers. see the copyright
// file at the top-level directory of this distribution and at
// http://rust-lang.org/copyright.
//
// licensed under the apache license, version 2.0 <license-apache or
// http://www.apache.org/licenses/license-2.0> or the mit license
// <license-mit or http://opensource.org/licenses/mit>, at your
// option. this file may not be copied, modified, or distributed
// except according to those terms.
use option::{Option, Some, None};
use result::{Ok, Err};
use rt::io::{io_error};
use rt::rtio::{IoFactory, IoFactoryObject,
               RtioTimer, RtioTimerObject};
use rt::local::Local;

pub struct Timer(~RtioTimerObject);

impl Timer {
    fn new(i: ~RtioTimerObject) -> Timer {
        Timer(i)
    }

    pub fn init() -> Option<Timer> {
        let timer = unsafe {
            rtdebug!("Timer::init: borrowing io to init timer");
            let io = Local::unsafe_borrow::<IoFactoryObject>();
            rtdebug!("about to init timer");
            (*io).timer_init()
        };
        match timer {
            Ok(t) => Some(Timer::new(t)),
            Err(ioerr) => {
                rtdebug!("Timer::init: failed to init: %?", ioerr);
                io_error::cond.raise(ioerr);
                None
            }
        }
    }
}

impl RtioTimer for Timer {
    fn sleep(&self, msecs: u64) {
        (**self).sleep(msecs);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rt::test::*;
    use option::{Some, None};
    #[test]
    fn test_io_timer_sleep_simple() {
        do run_in_newsched_task {
            let timer = Timer::init();
            match timer {
                Some(t) => t.sleep(1),
                None => assert!(false)
            }
        }
    }
}