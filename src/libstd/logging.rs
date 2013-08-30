// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Logging

use option::*;
use os;
use either::*;
use rt;
use rt::logging::{Logger, StdErrLogger};

/// Turns on logging to stdout globally
pub fn console_on() {
    rt::logging::console_on();
}

/**
 * Turns off logging to stdout globally
 *
 * Turns off the console unless the user has overridden the
 * runtime environment's logging spec, e.g. by setting
 * the RUST_LOG environment variable
 */
pub fn console_off() {
    // If RUST_LOG is set then the console can't be turned off
    if os::getenv("RUST_LOG").is_some() {
        return;
    }

    rt::logging::console_off();
}

#[cfg(not(test))]
#[lang="log_type"]
#[allow(missing_doc)]
pub fn log_type<T>(_level: u32, object: &T) {
    use sys;

    // XXX: Bad allocation
    let msg = sys::log_str(object);
    newsched_log_str(msg);
}

fn newsched_log_str(msg: ~str) {
    use rt::task::Task;
    use rt::local::Local;

    unsafe {
        let optional_task: Option<*mut Task> = Local::try_unsafe_borrow();
        match optional_task {
            Some(local) => {
                // Use the available logger
                (*local).logger.log(Left(msg));
            }
            None => {
                // There is no logger anywhere, just write to stderr
                let mut logger = StdErrLogger;
                logger.log(Left(msg));
            }
        }
    }
}
