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
use sys::unwind;
use sys::backtrace::Backtrace;
use sys::stdio;

use any::Any;
use cell::RefCell;
use sync::StaticMutex;
use backtrace;
use intrinsics;
use thread;
use fmt;
use io;

thread_local! {
    pub static LOCAL_STDERR: RefCell<Option<Box<Write + Send>>> = {
        RefCell::new(None)
    }
}

static BACKTRACE_LOCK: StaticMutex = StaticMutex::new();
static mut BACKTRACE: Backtrace = Backtrace::new();

fn log_panic(obj: &(Any+Send), file: &'static str, line: u32,
             log_backtrace: bool) {
    let msg = match obj.downcast_ref::<&'static str>() {
        Some(s) => *s,
        None => match obj.downcast_ref::<String>() {
            Some(s) => &s[..],
            None => "Box<Any>",
        }
    };
    let thread = thread::current();
    let name = thread.name().unwrap_or("<unnamed>");

    let write = |err: &mut Write| {
        let _ = writeln!(err, "thread '{}' panicked at '{}', {}:{}",
                         name, msg, file, line);
        if log_backtrace {
            let _g = BACKTRACE_LOCK.lock();
            let _ = unsafe { BACKTRACE.write(err) };
        }
    };

    if let Some(mut w) = LOCAL_STDERR.with(|s| s.borrow_mut().take()) {
        write(&mut *w);
        LOCAL_STDERR.with(move |slot| {
            *slot.borrow_mut() = Some(w);
        });
    } else {
        write(&mut io::stderr())
    }
}

pub fn on_panic(obj: &(Any+Send), file: &'static str, line: u32) {
    let panics = unwind::panic_inc();

    // If this is the third nested call, on_panic triggered the last panic,
    // otherwise the double-panic check would have aborted the process.
    // Even if it is likely that on_panic was unable to log the backtrace,
    // abort immediately to avoid infinite recursion, so that attaching a
    // debugger provides a useable stacktrace.
    if panics > 1 {
        stdio::dumb_print(format_args!("thread panicked while processing \
                                       panic. aborting."));
        unsafe { intrinsics::abort() }
    }

    // If this is a double panic, make sure that we print a backtrace
    // for this panic. Otherwise only print it if logging is enabled.
    let log_backtrace = panics > 0 || backtrace::log_enabled();
    log_panic(obj, file, line, log_backtrace);

    if panics > 0 {
        // If a thread panics while it's already unwinding then we
        // have limited options. Currently our preference is to
        // just abort. In the future we may consider resuming
        // unwinding or otherwise exiting the thread cleanly.
        stdio::dumb_print(format_args!("thread panicked while panicking. \
                                        aborting."));
        unsafe { intrinsics::abort() }
    }
}

pub fn abort(args: fmt::Arguments) -> ! {
    use intrinsics;

    stdio::dumb_print(format_args!("fatal runtime error: {}", args));
    unsafe { intrinsics::abort(); }
}

pub fn report_overflow() {
    stdio::dumb_print(format_args!("\nthread '{}' has overflowed its stack",
                                   thread::current().name().unwrap_or("<unknown>")));
}
