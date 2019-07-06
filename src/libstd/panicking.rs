//! Implementation of various bits and pieces of the `panic!` macro and
//! associated runtime pieces.
//!
//! Specifically, this module contains the implementation of:
//!
//! * Panic hooks
//! * Executing a panic up to doing the actual implementation
//! * Shims around "try"

use core::panic::{BoxMeUp, PanicInfo, Location};

use crate::any::Any;
use crate::fmt;
use crate::intrinsics;
use crate::mem;
use crate::ptr;
use crate::raw;
use crate::sys::stdio::panic_output;
use crate::sys_common::rwlock::RWLock;
use crate::sys_common::thread_info;
use crate::sys_common::util;
use crate::thread;

#[cfg(not(test))]
use crate::io::set_panic;
// make sure to use the stderr output configured
// by libtest in the real copy of std
#[cfg(test)]
use realstd::io::set_panic;

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
    #[unwind(allowed)]
    fn __rust_start_panic(payload: usize) -> u32;
}

#[derive(Copy, Clone)]
enum Hook {
    Default,
    Custom(*mut (dyn Fn(&PanicInfo<'_>) + 'static + Sync + Send)),
}

static HOOK_LOCK: RWLock = RWLock::new();
static mut HOOK: Hook = Hook::Default;

/// Registers a custom panic hook, replacing any that was previously registered.
///
/// The panic hook is invoked when a thread panics, but before the panic runtime
/// is invoked. As such, the hook will run with both the aborting and unwinding
/// runtimes. The default hook prints a message to standard error and generates
/// a backtrace if requested, but this behavior can be customized with the
/// `set_hook` and [`take_hook`] functions.
///
/// [`take_hook`]: ./fn.take_hook.html
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
///
/// # Examples
///
/// The following will print "Custom panic hook":
///
/// ```should_panic
/// use std::panic;
///
/// panic::set_hook(Box::new(|_| {
///     println!("Custom panic hook");
/// }));
///
/// panic!("Normal panic");
/// ```
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub fn set_hook(hook: Box<dyn Fn(&PanicInfo<'_>) + 'static + Sync + Send>) {
    if thread::panicking() {
        panic!("cannot modify the panic hook from a panicking thread");
    }

    unsafe {
        HOOK_LOCK.write();
        let old_hook = HOOK;
        HOOK = Hook::Custom(Box::into_raw(hook));
        HOOK_LOCK.write_unlock();

        if let Hook::Custom(ptr) = old_hook {
            #[allow(unused_must_use)] {
                Box::from_raw(ptr);
            }
        }
    }
}

/// Unregisters the current panic hook, returning it.
///
/// *See also the function [`set_hook`].*
///
/// [`set_hook`]: ./fn.set_hook.html
///
/// If no custom hook is registered, the default hook will be returned.
///
/// # Panics
///
/// Panics if called from a panicking thread.
///
/// # Examples
///
/// The following will print "Normal panic":
///
/// ```should_panic
/// use std::panic;
///
/// panic::set_hook(Box::new(|_| {
///     println!("Custom panic hook");
/// }));
///
/// let _ = panic::take_hook();
///
/// panic!("Normal panic");
/// ```
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub fn take_hook() -> Box<dyn Fn(&PanicInfo<'_>) + 'static + Sync + Send> {
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
            Hook::Custom(ptr) => Box::from_raw(ptr),
        }
    }
}

fn default_hook(info: &PanicInfo<'_>) {
    #[cfg(feature = "backtrace")]
    use crate::sys_common::backtrace;

    // If this is a double panic, make sure that we print a backtrace
    // for this panic. Otherwise only print it if logging is enabled.
    #[cfg(feature = "backtrace")]
    let log_backtrace = {
        let panics = update_panic_count(0);

        if panics >= 2 {
            Some(backtrace::PrintFormat::Full)
        } else {
            backtrace::log_enabled()
        }
    };

    // The current implementation always returns `Some`.
    let location = info.location().unwrap();

    let msg = match info.payload().downcast_ref::<&'static str>() {
        Some(s) => *s,
        None => match info.payload().downcast_ref::<String>() {
            Some(s) => &s[..],
            None => "Box<Any>",
        }
    };
    let thread = thread_info::current_thread();
    let name = thread.as_ref().and_then(|t| t.name()).unwrap_or("<unnamed>");

    let write = |err: &mut dyn crate::io::Write| {
        let _ = writeln!(err, "thread '{}' panicked at '{}', {}",
                         name, msg, location);

        #[cfg(feature = "backtrace")]
        {
            use crate::sync::atomic::{AtomicBool, Ordering};

            static FIRST_PANIC: AtomicBool = AtomicBool::new(true);

            if let Some(format) = log_backtrace {
                let _ = backtrace::print(err, format);
            } else if FIRST_PANIC.compare_and_swap(true, false, Ordering::SeqCst) {
                let _ = writeln!(err, "note: run with `RUST_BACKTRACE=1` \
                                       environment variable to display a backtrace.");
            }
        }
    };

    if let Some(mut local) = set_panic(None) {
        // NB. In `cfg(test)` this uses the forwarding impl
        // for `Box<dyn (::realstd::io::Write) + Send>`.
        write(&mut local);
        set_panic(Some(local));
    } else if let Some(mut out) = panic_output() {
        write(&mut out);
    }
}


#[cfg(not(test))]
#[doc(hidden)]
#[unstable(feature = "update_panic_count", issue = "0")]
pub fn update_panic_count(amt: isize) -> usize {
    use crate::cell::Cell;
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
pub unsafe fn r#try<R, F: FnOnce() -> R>(f: F) -> Result<R, Box<dyn Any + Send>> {
    #[allow(unions_with_drop_fields)]
    union Data<F, R> {
        f: F,
        r: R,
    }

    // We do some sketchy operations with ownership here for the sake of
    // performance. We can only  pass pointers down to
    // `__rust_maybe_catch_panic` (can't pass objects by value), so we do all
    // the ownership tracking here manually using a union.
    //
    // We go through a transition where:
    //
    // * First, we set the data to be the closure that we're going to call.
    // * When we make the function call, the `do_call` function below, we take
    //   ownership of the function pointer. At this point the `Data` union is
    //   entirely uninitialized.
    // * If the closure successfully returns, we write the return value into the
    //   data's return slot. Note that `ptr::write` is used as it's overwriting
    //   uninitialized data.
    // * Finally, when we come back out of the `__rust_maybe_catch_panic` we're
    //   in one of two states:
    //
    //      1. The closure didn't panic, in which case the return value was
    //         filled in. We move it out of `data` and return it.
    //      2. The closure panicked, in which case the return value wasn't
    //         filled in. In this case the entire `data` union is invalid, so
    //         there is no need to drop anything.
    //
    // Once we stack all that together we should have the "most efficient'
    // method of calling a catch panic whilst juggling ownership.
    let mut any_data = 0;
    let mut any_vtable = 0;
    let mut data = Data {
        f,
    };

    let r = __rust_maybe_catch_panic(do_call::<F, R>,
                                     &mut data as *mut _ as *mut u8,
                                     &mut any_data,
                                     &mut any_vtable);

    return if r == 0 {
        debug_assert!(update_panic_count(0) == 0);
        Ok(data.r)
    } else {
        update_panic_count(-1);
        debug_assert!(update_panic_count(0) == 0);
        Err(mem::transmute(raw::TraitObject {
            data: any_data as *mut _,
            vtable: any_vtable as *mut _,
        }))
    };

    fn do_call<F: FnOnce() -> R, R>(data: *mut u8) {
        unsafe {
            let data = data as *mut Data<F, R>;
            let f = ptr::read(&mut (*data).f);
            ptr::write(&mut (*data).r, f());
        }
    }
}

/// Determines whether the current thread is unwinding because of panic.
pub fn panicking() -> bool {
    update_panic_count(0) != 0
}

/// Entry point of panic from the libcore crate.
#[cfg(not(test))]
#[panic_handler]
#[unwind(allowed)]
pub fn rust_begin_panic(info: &PanicInfo<'_>) -> ! {
    continue_panic_fmt(&info)
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
#[cold]
// If panic_immediate_abort, inline the abort call,
// otherwise avoid inlining because of it is cold path.
#[cfg_attr(not(feature="panic_immediate_abort"),inline(never))]
#[cfg_attr(    feature="panic_immediate_abort" ,inline)]
pub fn begin_panic_fmt(msg: &fmt::Arguments<'_>,
                       file_line_col: &(&'static str, u32, u32)) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        unsafe { intrinsics::abort() }
    }

    let (file, line, col) = *file_line_col;
    let info = PanicInfo::internal_constructor(
        Some(msg),
        Location::internal_constructor(file, line, col),
    );
    continue_panic_fmt(&info)
}

fn continue_panic_fmt(info: &PanicInfo<'_>) -> ! {
    struct PanicPayload<'a> {
        inner: &'a fmt::Arguments<'a>,
        string: Option<String>,
    }

    impl<'a> PanicPayload<'a> {
        fn new(inner: &'a fmt::Arguments<'a>) -> PanicPayload<'a> {
            PanicPayload { inner, string: None }
        }

        fn fill(&mut self) -> &mut String {
            use crate::fmt::Write;

            let inner = self.inner;
            self.string.get_or_insert_with(|| {
                let mut s = String::new();
                drop(s.write_fmt(*inner));
                s
            })
        }
    }

    unsafe impl<'a> BoxMeUp for PanicPayload<'a> {
        fn box_me_up(&mut self) -> *mut (dyn Any + Send) {
            let contents = mem::take(self.fill());
            Box::into_raw(Box::new(contents))
        }

        fn get(&mut self) -> &(dyn Any + Send) {
            self.fill()
        }
    }

    // We do two allocations here, unfortunately. But (a) they're
    // required with the current scheme, and (b) we don't handle
    // panic + OOM properly anyway (see comment in begin_panic
    // below).

    let loc = info.location().unwrap(); // The current implementation always returns Some
    let msg = info.message().unwrap(); // The current implementation always returns Some
    let file_line_col = (loc.file(), loc.line(), loc.column());
    rust_panic_with_hook(
        &mut PanicPayload::new(msg),
        info.message(),
        &file_line_col);
}

/// This is the entry point of panicking for panic!() and assert!().
#[unstable(feature = "libstd_sys_internals",
           reason = "used by the panic! macro",
           issue = "0")]
#[cfg_attr(not(test), lang = "begin_panic")]
// never inline unless panic_immediate_abort to avoid code
// bloat at the call sites as much as possible
#[cfg_attr(not(feature="panic_immediate_abort"),inline(never))]
#[cold]
pub fn begin_panic<M: Any + Send>(msg: M, file_line_col: &(&'static str, u32, u32)) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        unsafe { intrinsics::abort() }
    }

    // Note that this should be the only allocation performed in this code path.
    // Currently this means that panic!() on OOM will invoke this code path,
    // but then again we're not really ready for panic on OOM anyway. If
    // we do start doing this, then we should propagate this allocation to
    // be performed in the parent of this thread instead of the thread that's
    // panicking.

    rust_panic_with_hook(&mut PanicPayload::new(msg), None, file_line_col);

    struct PanicPayload<A> {
        inner: Option<A>,
    }

    impl<A: Send + 'static> PanicPayload<A> {
        fn new(inner: A) -> PanicPayload<A> {
            PanicPayload { inner: Some(inner) }
        }
    }

    unsafe impl<A: Send + 'static> BoxMeUp for PanicPayload<A> {
        fn box_me_up(&mut self) -> *mut (dyn Any + Send) {
            let data = match self.inner.take() {
                Some(a) => Box::new(a) as Box<dyn Any + Send>,
                None => Box::new(()),
            };
            Box::into_raw(data)
        }

        fn get(&mut self) -> &(dyn Any + Send) {
            match self.inner {
                Some(ref a) => a,
                None => &(),
            }
        }
    }
}

/// Central point for dispatching panics.
///
/// Executes the primary logic for a panic, including checking for recursive
/// panics, panic hooks, and finally dispatching to the panic runtime to either
/// abort or unwind.
fn rust_panic_with_hook(payload: &mut dyn BoxMeUp,
                        message: Option<&fmt::Arguments<'_>>,
                        file_line_col: &(&str, u32, u32)) -> ! {
    let (file, line, col) = *file_line_col;

    let panics = update_panic_count(1);

    // If this is the third nested call (e.g., panics == 2, this is 0-indexed),
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
        let mut info = PanicInfo::internal_constructor(
            message,
            Location::internal_constructor(file, line, col),
        );
        HOOK_LOCK.read();
        match HOOK {
            // Some platforms know that printing to stderr won't ever actually
            // print anything, and if that's the case we can skip the default
            // hook.
            Hook::Default if panic_output().is_none() => {}
            Hook::Default => {
                info.set_payload(payload.get());
                default_hook(&info);
            }
            Hook::Custom(ptr) => {
                info.set_payload(payload.get());
                (*ptr)(&info);
            }
        };
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

    rust_panic(payload)
}

/// Shim around rust_panic. Called by resume_unwind.
pub fn update_count_then_panic(msg: Box<dyn Any + Send>) -> ! {
    update_panic_count(1);

    struct RewrapBox(Box<dyn Any + Send>);

    unsafe impl BoxMeUp for RewrapBox {
        fn box_me_up(&mut self) -> *mut (dyn Any + Send) {
            Box::into_raw(mem::replace(&mut self.0, Box::new(())))
        }

        fn get(&mut self) -> &(dyn Any + Send) {
            &*self.0
        }
    }

    rust_panic(&mut RewrapBox(msg))
}

/// An unmangled function (through `rustc_std_internal_symbol`) on which to slap
/// yer breakpoints.
#[inline(never)]
#[cfg_attr(not(test), rustc_std_internal_symbol)]
fn rust_panic(mut msg: &mut dyn BoxMeUp) -> ! {
    let code = unsafe {
        let obj = &mut msg as *mut &mut dyn BoxMeUp;
        __rust_start_panic(obj as usize)
    };
    rtabort!("failed to initiate panic, error {}", code)
}
