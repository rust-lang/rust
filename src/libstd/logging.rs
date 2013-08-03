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
use rt::OldTaskContext;
use rt::logging::{Logger, StdErrLogger};

/// Turns on logging to stdout globally
pub fn console_on() {
    if rt::context() == OldTaskContext {
        unsafe {
            rustrt::rust_log_console_on();
        }
    } else {
        rt::logging::console_on();
    }
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

    if rt::context() == OldTaskContext {
        unsafe {
            rustrt::rust_log_console_off();
        }
    } else {
        rt::logging::console_off();
    }
}

#[cfg(not(test))]
#[lang="log_type"]
#[allow(missing_doc)]
pub fn log_type<T>(level: u32, object: &T) {
    use cast;
    use container::Container;
    use io;
    use libc;
    use repr;
    use rt;
    use str;
    use vec;

    let bytes = do io::with_bytes_writer |writer| {
        repr::write_repr(writer, object);
    };

    match rt::context() {
        rt::OldTaskContext => {
            unsafe {
                let len = bytes.len() as libc::size_t;
                rustrt::rust_log_str(level, cast::transmute(vec::raw::to_ptr(bytes)), len);
            }
        }
        _ => {
            // XXX: Bad allocation
            let msg = str::from_bytes(bytes);
            newsched_log_str(msg);
        }
    }
}

fn newsched_log_str(msg: ~str) {
    use rt::task::Task;
    use rt::local::Local;

    unsafe {
        match Local::try_unsafe_borrow::<Task>() {
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

pub mod rustrt {
    use libc;

    extern {
        pub fn rust_log_console_on();
        pub fn rust_log_console_off();
        pub fn rust_log_str(level: u32,
                            string: *libc::c_char,
                            size: libc::size_t);
    }
}
