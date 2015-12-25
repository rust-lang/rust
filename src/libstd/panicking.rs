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
use sync::StaticRwLock;
use sys::stdio::Stderr;
use sys_common::backtrace;
use sys_common::thread_info;
use sys_common::util;
use thread;

thread_local! { pub static PANIC_COUNT: Cell<usize> = Cell::new(0) }

thread_local! {
    pub static LOCAL_STDERR: RefCell<Option<Box<Write + Send>>> = {
        RefCell::new(None)
    }
}

#[derive(Copy, Clone)]
enum Handler {
    Default,
    Custom(*mut (Fn(&PanicInfo) + 'static + Sync + Send)),
}

static HANDLER_LOCK: StaticRwLock = StaticRwLock::new();
static mut HANDLER: Handler = Handler::Default;

/// Registers a custom panic handler, replacing any that was previously
/// registered.
///
/// The panic handler is invoked when a thread panics, but before it begins
/// unwinding the stack. The default handler prints a message to standard error
/// and generates a backtrace if requested, but this behavior can be customized
/// with the `set_handler` and `take_handler` functions.
///
/// The handler is provided with a `PanicInfo` struct which contains information
/// about the origin of the panic, including the payload passed to `panic!` and
/// the source code location from which the panic originated.
///
/// The panic handler is a global resource.
///
/// # Panics
///
/// Panics if called from a panicking thread.
#[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
pub fn set_handler<F>(handler: F) where F: Fn(&PanicInfo) + 'static + Sync + Send {
    if thread::panicking() {
        panic!("cannot modify the panic handler from a panicking thread");
    }

    let handler = Box::new(handler);
    unsafe {
        let lock = HANDLER_LOCK.write();
        let old_handler = HANDLER;
        HANDLER = Handler::Custom(Box::into_raw(handler));
        drop(lock);

        if let Handler::Custom(ptr) = old_handler {
            Box::from_raw(ptr);
        }
    }
}

/// Unregisters the current panic handler, returning it.
///
/// If no custom handler is registered, the default handler will be returned.
///
/// # Panics
///
/// Panics if called from a panicking thread.
#[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
pub fn take_handler() -> Box<Fn(&PanicInfo) + 'static + Sync + Send> {
    if thread::panicking() {
        panic!("cannot modify the panic handler from a panicking thread");
    }

    unsafe {
        let lock = HANDLER_LOCK.write();
        let handler = HANDLER;
        HANDLER = Handler::Default;
        drop(lock);

        match handler {
            Handler::Default => Box::new(default_handler),
            Handler::Custom(ptr) => {Box::from_raw(ptr)} // FIXME #30530
        }
    }
}

/// A struct providing information about a panic.
#[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
pub struct PanicInfo<'a> {
    payload: &'a (Any + Send),
    location: Location<'a>,
}

impl<'a> PanicInfo<'a> {
    /// Returns the payload associated with the panic.
    ///
    /// This will commonly, but not always, be a `&'static str` or `String`.
    #[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
    pub fn payload(&self) -> &(Any + Send) {
        self.payload
    }

    /// Returns information about the location from which the panic originated,
    /// if available.
    ///
    /// This method will currently always return `Some`, but this may change
    /// in future versions.
    #[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
    pub fn location(&self) -> Option<&Location> {
        Some(&self.location)
    }
}

/// A struct containing information about the location of a panic.
#[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
pub struct Location<'a> {
    file: &'a str,
    line: u32,
}

impl<'a> Location<'a> {
    /// Returns the name of the source file from which the panic originated.
    #[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
    pub fn file(&self) -> &str {
        self.file
    }

    /// Returns the line number from which the panic originated.
    #[unstable(feature = "panic_handler", reason = "awaiting feedback", issue = "30449")]
    pub fn line(&self) -> u32 {
        self.line
    }
}

fn default_handler(info: &PanicInfo) {
    let panics = PANIC_COUNT.with(|s| s.get());

    // If this is a double panic, make sure that we print a backtrace
    // for this panic. Otherwise only print it if logging is enabled.
    let log_backtrace = panics >= 2 || backtrace::log_enabled();

    let file = info.location.file;
    let line = info.location.line;

    let msg = match info.payload.downcast_ref::<&'static str>() {
        Some(s) => *s,
        None => match info.payload.downcast_ref::<String>() {
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

    let info = PanicInfo {
        payload: obj,
        location: Location {
            file: file,
            line: line,
        },
    };

    unsafe {
        let _lock = HANDLER_LOCK.read();
        match HANDLER {
            Handler::Default => default_handler(&info),
            Handler::Custom(ptr) => (*ptr)(&info),
        }
    }

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
