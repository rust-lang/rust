// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;
use io::prelude::*;

use any::Any;
use cell::Cell;
use cell::RefCell;
use intrinsics;
use sys::stdio::Stderr;
use sys_common::backtrace;
use sys_common::thread_info;
use sys_common::util;

thread_local! { pub static PANIC_COUNT: Cell<usize> = Cell::new(0) }

thread_local! {
    pub static LOCAL_STDERR: RefCell<Option<Box<Write + Send>>> = {
        RefCell::new(None)
    }
}

fn log_panic(obj: &(Any+Send), file: &'static str, line: u32,
             log_backtrace: bool) {
    let msg = match obj.downcast_ref::<&'static str>() {
        Some(s) => *s,
        None => match obj.downcast_ref::<String>() {
            Some(s) => &s[..],
            None => "Box<Any>",
        }
    };
    let mut err = Stderr::new().ok();
    let thread = thread_info::current_thread();
    let name = thread.as_ref().and_then(|t| t.name()).unwrap_or("<unnamed>");

    let write = |err: &mut ::io::Write| {
        let _ = writeln!(err, "thread '{}' panicked at '{}', {}:{}",
                         name, msg, file, line);
        if log_backtrace {
            let _ = backtrace::write(err);
        }
    };

    let prev = LOCAL_STDERR.with(|s| s.borrow_mut().take());
    match (prev, err.as_mut()) {
        (Some(mut stderr), _) => {
            write(&mut *stderr);
            let mut s = Some(stderr);
            LOCAL_STDERR.with(|slot| {
                *slot.borrow_mut() = s.take();
            });
        }
        (None, Some(ref mut err)) => { write(err) }
        _ => {}
    }
}

pub fn on_panic(obj: &(Any+Send), file: &'static str, line: u32) {
    let panics = PANIC_COUNT.with(|s| {
        let count = s.get() + 1;
        s.set(count);
        count
    });

    // If this is the third nested call, on_panic triggered the last panic,
    // otherwise the double-panic check would have aborted the process.
    // Even if it is likely that on_panic was unable to log the backtrace,
    // abort immediately to avoid infinite recursion, so that attaching a
    // debugger provides a useable stacktrace.
    if panics >= 3 {
        util::dumb_print(format_args!("thread panicked while processing \
                                       panic. aborting."));
        unsafe { intrinsics::abort() }
    }

    // If this is a double panic, make sure that we print a backtrace
    // for this panic. Otherwise only print it if logging is enabled.
    let log_backtrace = panics >= 2 || backtrace::log_enabled();
    log_panic(obj, file, line, log_backtrace);

    if panics >= 2 {
        // If a thread panics while it's already unwinding then we
        // have limited options. Currently our preference is to
        // just abort. In the future we may consider resuming
        // unwinding or otherwise exiting the thread cleanly.
        util::dumb_print(format_args!("thread panicked while panicking. \
                                       aborting."));
        unsafe { intrinsics::abort() }
    }
}
