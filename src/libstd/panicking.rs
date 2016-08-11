// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of various bits and pieces of the `panic!` macro and
//! associated runtime pieces.
//!
//! Specifically, this module contains the implementation of:
//!
//! * Panic hooks
//! * Executing a panic up to doing the actual implementation
//! * Shims around "try"

use prelude::v1::*;
use io::prelude::*;

use any::Any;
use cell::RefCell;
use fmt;
use intrinsics;
use mem;
use raw;
use sys_common::rwlock::RWLock;
use sys::stdio::Stderr;
use sys_common::thread_info;
use sys_common::util;
use thread;

thread_local! {
    pub static LOCAL_STDERR: RefCell<Option<Box<Write + Send>>> = {
        RefCell::new(None)
    }
}

// Binary interface to the panic runtime that the standard library depends on.
//
// The standard library is tagged with `#![needs_panic_runtime]` (introduced in
// RFC 1513) to indicate that it requires some other crate tagged with
// `#![panic_runtime]` to exist somewhere. Each panic runtime is intended to
// implement these symbols (with the same signatures) so we can get matched up
// to them.
//
// One day this may look a little less ad-hoc with the compiler helping out to
// hook up these functions, but it is not this day!
#[allow(improper_ctypes)]
extern {
    fn __rust_maybe_catch_panic(f: fn(*mut u8),
                                data: *mut u8,
                                data_ptr: *mut usize,
                                vtable_ptr: *mut usize) -> u32;
    #[unwind]
    fn __rust_start_panic(data: usize, vtable: usize) -> u32;
}

#[derive(Copy, Clone)]
enum Hook {
    Default,
    Custom(*mut (Fn(&PanicInfo) + 'static + Sync + Send)),
}

static HOOK_LOCK: RWLock = RWLock::new();
static mut HOOK: Hook = Hook::Default;

/// Registers a custom panic hook, replacing any that was previously registered.
///
/// The panic hook is invoked when a thread panics, but before the panic runtime
/// is invoked. As such, the hook will run with both the aborting and unwinding
/// runtimes. The default hook prints a message to standard error and generates
/// a backtrace if requested, but this behavior can be customized with the
/// `set_hook` and `take_hook` functions.
///
/// The hook is provided with a `PanicInfo` struct which contains information
/// about the origin of the panic, including the payload passed to `panic!` and
/// the source code location from which the panic originated.
///
/// The panic hook is a global resource.
///
/// # Panics
///
/// Panics if called from a panicking thread.
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub fn set_hook(hook: Box<Fn(&PanicInfo) + 'static + Sync + Send>) {
    if thread::panicking() {
        panic!("cannot modify the panic hook from a panicking thread");
    }

    unsafe {
        HOOK_LOCK.write();
        let old_hook = HOOK;
        HOOK = Hook::Custom(Box::into_raw(hook));
        HOOK_LOCK.write_unlock();

        if let Hook::Custom(ptr) = old_hook {
            Box::from_raw(ptr);
        }
    }
}

/// Unregisters the current panic hook, returning it.
///
/// If no custom hook is registered, the default hook will be returned.
///
/// # Panics
///
/// Panics if called from a panicking thread.
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub fn take_hook() -> Box<Fn(&PanicInfo) + 'static + Sync + Send> {
    if thread::panicking() {
        panic!("cannot modify the panic hook from a panicking thread");
    }

    unsafe {
        HOOK_LOCK.write();
        let hook = HOOK;
        HOOK = Hook::Default;
        HOOK_LOCK.write_unlock();

        match hook {
            Hook::Default => Box::new(default_hook),
            Hook::Custom(ptr) => {Box::from_raw(ptr)} // FIXME #30530
        }
    }
}

/// A struct providing information about a panic.
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub struct PanicInfo<'a> {
    payload: &'a (Any + Send),
    location: Location<'a>,
}

impl<'a> PanicInfo<'a> {
    /// Returns the payload associated with the panic.
    ///
    /// This will commonly, but not always, be a `&'static str` or `String`.
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn payload(&self) -> &(Any + Send) {
        self.payload
    }

    /// Returns information about the location from which the panic originated,
    /// if available.
    ///
    /// This method will currently always return `Some`, but this may change
    /// in future versions.
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn location(&self) -> Option<&Location> {
        Some(&self.location)
    }
}

/// A struct containing information about the location of a panic.
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub struct Location<'a> {
    file: &'a str,
    line: u32,
}

impl<'a> Location<'a> {
    /// Returns the name of the source file from which the panic originated.
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn file(&self) -> &str {
        self.file
    }

    /// Returns the line number from which the panic originated.
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn line(&self) -> u32 {
        self.line
    }
}

fn default_hook(info: &PanicInfo) {
    #[cfg(any(not(cargobuild), feature = "backtrace"))]
    use sys_common::backtrace;

    // If this is a double panic, make sure that we print a backtrace
    // for this panic. Otherwise only print it if logging is enabled.
    #[cfg(any(not(cargobuild), feature = "backtrace"))]
    let log_backtrace = {
        let panics = update_panic_count(0);

        panics >= 2 || backtrace::log_enabled()
    };

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

        #[cfg(any(not(cargobuild), feature = "backtrace"))]
        {
            use sync::atomic::{AtomicBool, Ordering};

            static FIRST_PANIC: AtomicBool = AtomicBool::new(true);

            if log_backtrace {
                let _ = backtrace::write(err);
            } else if FIRST_PANIC.compare_and_swap(true, false, Ordering::SeqCst) {
                let _ = writeln!(err, "note: Run with `RUST_BACKTRACE=1` for a backtrace.");
            }
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


#[cfg(not(test))]
#[doc(hidden)]
#[unstable(feature = "update_panic_count", issue = "0")]
pub fn update_panic_count(amt: isize) -> usize {
    use cell::Cell;
    thread_local! { static PANIC_COUNT: Cell<usize> = Cell::new(0) }

    PANIC_COUNT.with(|c| {
        let next = (c.get() as isize + amt) as usize;
        c.set(next);
        return next
    })
}

#[cfg(test)]
pub use realstd::rt::update_panic_count;

/// Invoke a closure, capturing the cause of an unwinding panic if one occurs.
pub unsafe fn try<R, F: FnOnce() -> R>(f: F) -> Result<R, Box<Any + Send>> {
    let mut slot = None;
    let mut f = Some(f);
    let ret;

    {
        let mut to_run = || {
            slot = Some(f.take().unwrap()());
        };
        let fnptr = get_call(&mut to_run);
        let dataptr = &mut to_run as *mut _ as *mut u8;
        let mut any_data = 0;
        let mut any_vtable = 0;
        let fnptr = mem::transmute::<fn(&mut _), fn(*mut u8)>(fnptr);
        let r = __rust_maybe_catch_panic(fnptr,
                                         dataptr,
                                         &mut any_data,
                                         &mut any_vtable);
        if r == 0 {
            ret = Ok(());
        } else {
            update_panic_count(-1);
            ret = Err(mem::transmute(raw::TraitObject {
                data: any_data as *mut _,
                vtable: any_vtable as *mut _,
            }));
        }
    }

    debug_assert!(update_panic_count(0) == 0);
    return ret.map(|()| {
        slot.take().unwrap()
    });

    fn get_call<F: FnMut()>(_: &mut F) -> fn(&mut F) {
        call
    }

    fn call<F: FnMut()>(f: &mut F) {
        f()
    }
}

/// Determines whether the current thread is unwinding because of panic.
pub fn panicking() -> bool {
    update_panic_count(0) != 0
}

/// Entry point of panic from the libcore crate.
#[cfg(not(test))]
#[lang = "panic_fmt"]
#[unwind]
pub extern fn rust_begin_panic(msg: fmt::Arguments,
                               file: &'static str,
                               line: u32) -> ! {
    begin_panic_fmt(&msg, &(file, line))
}

/// The entry point for panicking with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `panic!()` has as low an impact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[unstable(feature = "libstd_sys_internals",
           reason = "used by the panic! macro",
           issue = "0")]
#[inline(never)] #[cold]
pub fn begin_panic_fmt(msg: &fmt::Arguments,
                       file_line: &(&'static str, u32)) -> ! {
    use fmt::Write;

    // We do two allocations here, unfortunately. But (a) they're
    // required with the current scheme, and (b) we don't handle
    // panic + OOM properly anyway (see comment in begin_panic
    // below).

    let mut s = String::new();
    let _ = s.write_fmt(*msg);
    begin_panic(s, file_line)
}

/// This is the entry point of panicking for panic!() and assert!().
#[unstable(feature = "libstd_sys_internals",
           reason = "used by the panic! macro",
           issue = "0")]
#[inline(never)] #[cold] // avoid code bloat at the call sites as much as possible
pub fn begin_panic<M: Any + Send>(msg: M, file_line: &(&'static str, u32)) -> ! {
    // Note that this should be the only allocation performed in this code path.
    // Currently this means that panic!() on OOM will invoke this code path,
    // but then again we're not really ready for panic on OOM anyway. If
    // we do start doing this, then we should propagate this allocation to
    // be performed in the parent of this thread instead of the thread that's
    // panicking.

    rust_panic_with_hook(Box::new(msg), file_line)
}

/// Executes the primary logic for a panic, including checking for recursive
/// panics and panic hooks.
///
/// This is the entry point or panics from libcore, formatted panics, and
/// `Box<Any>` panics. Here we'll verify that we're not panicking recursively,
/// run panic hooks, and then delegate to the actual implementation of panics.
#[inline(never)]
#[cold]
fn rust_panic_with_hook(msg: Box<Any + Send>,
                        file_line: &(&'static str, u32)) -> ! {
    let (file, line) = *file_line;

    let panics = update_panic_count(1);

    // If this is the third nested call (e.g. panics == 2, this is 0-indexed),
    // the panic hook probably triggered the last panic, otherwise the
    // double-panic check would have aborted the process. In this case abort the
    // process real quickly as we don't want to try calling it again as it'll
    // probably just panic again.
    if panics > 2 {
        util::dumb_print(format_args!("thread panicked while processing \
                                       panic. aborting.\n"));
        unsafe { intrinsics::abort() }
    }

    unsafe {
        let info = PanicInfo {
            payload: &*msg,
            location: Location {
                file: file,
                line: line,
            },
        };
        HOOK_LOCK.read();
        match HOOK {
            Hook::Default => default_hook(&info),
            Hook::Custom(ptr) => (*ptr)(&info),
        }
        HOOK_LOCK.read_unlock();
    }

    if panics > 1 {
        // If a thread panics while it's already unwinding then we
        // have limited options. Currently our preference is to
        // just abort. In the future we may consider resuming
        // unwinding or otherwise exiting the thread cleanly.
        util::dumb_print(format_args!("thread panicked while panicking. \
                                       aborting.\n"));
        unsafe { intrinsics::abort() }
    }

    rust_panic(msg)
}

/// Shim around rust_panic. Called by resume_unwind.
pub fn update_count_then_panic(msg: Box<Any + Send>) -> ! {
    update_panic_count(1);
    rust_panic(msg)
}

/// A private no-mangle function on which to slap yer breakpoints.
#[no_mangle]
#[allow(private_no_mangle_fns)] // yes we get it, but we like breakpoints
pub fn rust_panic(msg: Box<Any + Send>) -> ! {
    let code = unsafe {
        let obj = mem::transmute::<_, raw::TraitObject>(msg);
        __rust_start_panic(obj.data as usize, obj.vtable as usize)
    };
    rtabort!("failed to initiate panic, error {}", code)
}
