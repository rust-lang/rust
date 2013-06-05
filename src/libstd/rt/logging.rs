// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use either::*;

pub trait Logger {
    fn log(&mut self, msg: Either<~str, &'static str>);
}

pub struct StdErrLogger;

impl Logger for StdErrLogger {
    fn log(&mut self, msg: Either<~str, &'static str>) {
        use io::{Writer, WriterUtil};

        let s: &str = match msg {
            Left(ref s) => {
                let s: &str = *s;
                s
            }
            Right(ref s) => {
                let s: &str = *s;
                s
            }
        };
        let dbg = ::libc::STDERR_FILENO as ::io::fd_t;
        dbg.write_str(s);
        dbg.write_str("\n");
        dbg.flush();
    }
}

/// Configure logging by traversing the crate map and setting the
/// per-module global logging flags based on the logging spec
pub fn init(crate_map: *u8) {
    use os;
    use str;
    use ptr;
    use option::{Some, None};
    use libc::c_char;

    let log_spec = os::getenv("RUST_LOG");
    match log_spec {
        Some(spec) => {
            do str::as_c_str(spec) |s| {
                unsafe {
                    rust_update_log_settings(crate_map, s);
                }
            }
        }
        None => {
            unsafe {
                rust_update_log_settings(crate_map, ptr::null());
            }
        }
    }

    extern {
        fn rust_update_log_settings(crate_map: *u8, settings: *c_char);
    }
}
