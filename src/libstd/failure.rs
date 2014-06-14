// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::owned::Box;
use any::{Any, AnyRefExt};
use fmt;
use io::{Writer, IoResult};
use kinds::Send;
use option::{Some, None};
use result::Ok;
use rt::backtrace;
use rt::{Stderr, Stdio};
use rustrt::local::Local;
use rustrt::task::Task;
use str::Str;
use string::String;

// Defined in this module instead of io::stdio so that the unwinding
local_data_key!(pub local_stderr: Box<Writer + Send>)

impl Writer for Stdio {
    fn write(&mut self, bytes: &[u8]) -> IoResult<()> {
        fn fmt_write<F: fmt::FormatWriter>(f: &mut F, bytes: &[u8]) {
            let _ = f.write(bytes);
        }
        fmt_write(self, bytes);
        Ok(())
    }
}

pub fn on_fail(obj: &Any + Send, file: &'static str, line: uint) {
    let msg = match obj.as_ref::<&'static str>() {
        Some(s) => *s,
        None => match obj.as_ref::<String>() {
            Some(s) => s.as_slice(),
            None => "Box<Any>",
        }
    };
    let mut err = Stderr;

    // It is assumed that all reasonable rust code will have a local task at
    // all times. This means that this `exists` will return true almost all of
    // the time. There are border cases, however, when the runtime has
    // *almost* set up the local task, but hasn't quite gotten there yet. In
    // order to get some better diagnostics, we print on failure and
    // immediately abort the whole process if there is no local task
    // available.
    if !Local::exists(None::<Task>) {
        let _ = writeln!(&mut err, "failed at '{}', {}:{}", msg, file, line);
        if backtrace::log_enabled() {
            let _ = backtrace::write(&mut err);
        } else {
            let _ = writeln!(&mut err, "run with `RUST_BACKTRACE=1` to \
                                        see a backtrace");
        }
        return
    }

    // Peel the name out of local task so we can print it. We've got to be sure
    // that the local task is in TLS while we're printing as I/O may occur.
    let (name, unwinding) = {
        let mut t = Local::borrow(None::<Task>);
        (t.name.take(), t.unwinder.unwinding())
    };
    {
        let n = name.as_ref().map(|n| n.as_slice()).unwrap_or("<unnamed>");

        match local_stderr.replace(None) {
            Some(mut stderr) => {
                // FIXME: what to do when the task printing fails?
                let _ = writeln!(stderr,
                                 "task '{}' failed at '{}', {}:{}\n",
                                 n, msg, file, line);
                if backtrace::log_enabled() {
                    let _ = backtrace::write(stderr);
                }
                local_stderr.replace(Some(stderr));
            }
            None => {
                let _ = writeln!(&mut err, "task '{}' failed at '{}', {}:{}",
                                 n, msg, file, line);
                if backtrace::log_enabled() {
                    let _ = backtrace::write(&mut err);
                }
            }
        }

        // If this is a double failure, make sure that we printed a backtrace
        // for this failure.
        if unwinding && !backtrace::log_enabled() {
            let _ = backtrace::write(&mut err);
        }
    }
    Local::borrow(None::<Task>).name = name;
}
