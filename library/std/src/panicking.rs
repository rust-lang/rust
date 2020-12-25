//! Implementation of various bits and pieces of the `panic!` macro and
//! associated runtime pieces.
//!
//! Specifically, this module contains the implementation of:
//!
//! * Panic hooks
//! * Executing a panic up to doing the actual implementation
//! * Shims around "try"

#![deny(unsafe_op_in_unsafe_fn)]

use core::panic::{BoxMeUp, Location, PanicInfo};

use crate::any::Any;
use crate::fmt;
use crate::intrinsics;
use crate::mem::{self, ManuallyDrop};
use crate::process;
use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sys::stdio::panic_output;
use crate::sys_common::backtrace::{self, RustBacktrace};
use crate::sys_common::rwlock::RWLock;
use crate::sys_common::{thread_info, util};
use crate::thread;

#[cfg(not(test))]
use crate::io::set_output_capture;
// make sure to use the stderr output configured
// by libtest in the real copy of std
#[cfg(test)]
use realstd::io::set_output_capture;

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
extern "C" {
    fn __rust_panic_cleanup(payload: *mut u8) -> *mut (dyn Any + Send + 'static);

    /// `payload` is passed through another layer of raw pointers as `&mut dyn Trait` is not
    /// FFI-safe. `BoxMeUp` lazily performs allocation only when needed (this avoids allocations
    /// when using the "abort" panic runtime).
    #[unwind(allowed)]
    fn __rust_start_panic(payload: *mut &mut dyn BoxMeUp) -> u32;
}

/// This function is called by the panic runtime if FFI code catches a Rust
/// panic but doesn't rethrow it. We don't support this case since it messes
/// with our panic count.
#[cfg(not(test))]
#[rustc_std_internal_symbol]
extern "C" fn __rust_drop_panic() -> ! {
    rtabort!("Rust panics must be rethrown");
}

/// This function is called by the panic runtime if it catches an exception
/// object which does not correspond to a Rust panic.
#[cfg(not(test))]
#[rustc_std_internal_symbol]
extern "C" fn __rust_foreign_exception() -> ! {
    rtabort!("Rust cannot catch foreign exceptions");
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
            #[allow(unused_must_use)]
            {
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
    // If this is a double panic, make sure that we print a backtrace
    // for this panic. Otherwise only print it if logging is enabled.
    let backtrace_env = if panic_count::get() >= 2 {
        RustBacktrace::Print(crate::backtrace_rs::PrintFmt::Full)
    } else {
        backtrace::rust_backtrace_env()
    };

    // The current implementation always returns `Some`.
    let location = info.location().unwrap();

    let msg = match info.payload().downcast_ref::<&'static str>() {
        Some(s) => *s,
        None => match info.payload().downcast_ref::<String>() {
            Some(s) => &s[..],
            None => "Box<Any>",
        },
    };
    let thread = thread_info::current_thread();
    let name = thread.as_ref().and_then(|t| t.name()).unwrap_or("<unnamed>");

    let write = |err: &mut dyn crate::io::Write| {
        let _ = writeln!(err, "thread '{}' panicked at '{}', {}", name, msg, location);

        static FIRST_PANIC: AtomicBool = AtomicBool::new(true);

        match backtrace_env {
            RustBacktrace::Print(format) => drop(backtrace::print(err, format)),
            RustBacktrace::Disabled => {}
            RustBacktrace::RuntimeDisabled => {
                if FIRST_PANIC.swap(false, Ordering::SeqCst) {
                    let _ = writeln!(
                        err,
                        "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace"
                    );
                }
            }
        }
    };

    if let Some(local) = set_output_capture(None) {
        write(&mut *local.lock().unwrap_or_else(|e| e.into_inner()));
        set_output_capture(Some(local));
    } else if let Some(mut out) = panic_output() {
        write(&mut out);
    }
}

#[cfg(not(test))]
#[doc(hidden)]
#[unstable(feature = "update_panic_count", issue = "none")]
pub mod panic_count {
    use crate::cell::Cell;
    use crate::sync::atomic::{AtomicUsize, Ordering};

    // Panic count for the current thread.
    thread_local! { static LOCAL_PANIC_COUNT: Cell<usize> = Cell::new(0) }

    // Sum of panic counts from all threads. The purpose of this is to have
    // a fast path in `is_zero` (which is used by `panicking`). In any particular
    // thread, if that thread currently views `GLOBAL_PANIC_COUNT` as being zero,
    // then `LOCAL_PANIC_COUNT` in that thread is zero. This invariant holds before
    // and after increase and decrease, but not necessarily during their execution.
    static GLOBAL_PANIC_COUNT: AtomicUsize = AtomicUsize::new(0);

    pub fn increase() -> usize {
        GLOBAL_PANIC_COUNT.fetch_add(1, Ordering::Relaxed);
        LOCAL_PANIC_COUNT.with(|c| {
            let next = c.get() + 1;
            c.set(next);
            next
        })
    }

    pub fn decrease() -> usize {
        GLOBAL_PANIC_COUNT.fetch_sub(1, Ordering::Relaxed);
        LOCAL_PANIC_COUNT.with(|c| {
            let next = c.get() - 1;
            c.set(next);
            next
        })
    }

    pub fn get() -> usize {
        LOCAL_PANIC_COUNT.with(|c| c.get())
    }

    #[inline]
    pub fn is_zero() -> bool {
        if GLOBAL_PANIC_COUNT.load(Ordering::Relaxed) == 0 {
            // Fast path: if `GLOBAL_PANIC_COUNT` is zero, all threads
            // (including the current one) will have `LOCAL_PANIC_COUNT`
            // equal to zero, so TLS access can be avoided.
            //
            // In terms of performance, a relaxed atomic load is similar to a normal
            // aligned memory read (e.g., a mov instruction in x86), but with some
            // compiler optimization restrictions. On the other hand, a TLS access
            // might require calling a non-inlinable function (such as `__tls_get_addr`
            // when using the GD TLS model).
            true
        } else {
            is_zero_slow_path()
        }
    }

    // Slow path is in a separate function to reduce the amount of code
    // inlined from `is_zero`.
    #[inline(never)]
    #[cold]
    fn is_zero_slow_path() -> bool {
        LOCAL_PANIC_COUNT.with(|c| c.get() == 0)
    }
}

#[cfg(test)]
pub use realstd::rt::panic_count;

/// Invoke a closure, capturing the cause of an unwinding panic if one occurs.
pub unsafe fn r#try<R, F: FnOnce() -> R>(f: F) -> Result<R, Box<dyn Any + Send>> {
    union Data<F, R> {
        f: ManuallyDrop<F>,
        r: ManuallyDrop<R>,
        p: ManuallyDrop<Box<dyn Any + Send>>,
    }

    // We do some sketchy operations with ownership here for the sake of
    // performance. We can only pass pointers down to `do_call` (can't pass
    // objects by value), so we do all the ownership tracking here manually
    // using a union.
    //
    // We go through a transition where:
    //
    // * First, we set the data field `f` to be the argumentless closure that we're going to call.
    // * When we make the function call, the `do_call` function below, we take
    //   ownership of the function pointer. At this point the `data` union is
    //   entirely uninitialized.
    // * If the closure successfully returns, we write the return value into the
    //   data's return slot (field `r`).
    // * If the closure panics (`do_catch` below), we write the panic payload into field `p`.
    // * Finally, when we come back out of the `try` intrinsic we're
    //   in one of two states:
    //
    //      1. The closure didn't panic, in which case the return value was
    //         filled in. We move it out of `data.r` and return it.
    //      2. The closure panicked, in which case the panic payload was
    //         filled in. We move it out of `data.p` and return it.
    //
    // Once we stack all that together we should have the "most efficient'
    // method of calling a catch panic whilst juggling ownership.
    let mut data = Data { f: ManuallyDrop::new(f) };

    let data_ptr = &mut data as *mut _ as *mut u8;
    // SAFETY:
    //
    // Access to the union's fields: this is `std` and we know that the `r#try`
    // intrinsic fills in the `r` or `p` union field based on its return value.
    //
    // The call to `intrinsics::r#try` is made safe by:
    // - `do_call`, the first argument, can be called with the initial `data_ptr`.
    // - `do_catch`, the second argument, can be called with the `data_ptr` as well.
    // See their safety preconditions for more informations
    unsafe {
        return if intrinsics::r#try(do_call::<F, R>, data_ptr, do_catch::<F, R>) == 0 {
            Ok(ManuallyDrop::into_inner(data.r))
        } else {
            Err(ManuallyDrop::into_inner(data.p))
        };
    }

    // We consider unwinding to be rare, so mark this function as cold. However,
    // do not mark it no-inline -- that decision is best to leave to the
    // optimizer (in most cases this function is not inlined even as a normal,
    // non-cold function, though, as of the writing of this comment).
    #[cold]
    unsafe fn cleanup(payload: *mut u8) -> Box<dyn Any + Send + 'static> {
        // SAFETY: The whole unsafe block hinges on a correct implementation of
        // the panic handler `__rust_panic_cleanup`. As such we can only
        // assume it returns the correct thing for `Box::from_raw` to work
        // without undefined behavior.
        let obj = unsafe { Box::from_raw(__rust_panic_cleanup(payload)) };
        panic_count::decrease();
        obj
    }

    // SAFETY:
    // data must be non-NUL, correctly aligned, and a pointer to a `Data<F, R>`
    // Its must contains a valid `f` (type: F) value that can be use to fill
    // `data.r`.
    //
    // This function cannot be marked as `unsafe` because `intrinsics::r#try`
    // expects normal function pointers.
    #[inline]
    fn do_call<F: FnOnce() -> R, R>(data: *mut u8) {
        // SAFETY: this is the responsibilty of the caller, see above.
        unsafe {
            let data = data as *mut Data<F, R>;
            let data = &mut (*data);
            let f = ManuallyDrop::take(&mut data.f);
            data.r = ManuallyDrop::new(f());
        }
    }

    // We *do* want this part of the catch to be inlined: this allows the
    // compiler to properly track accesses to the Data union and optimize it
    // away most of the time.
    //
    // SAFETY:
    // data must be non-NUL, correctly aligned, and a pointer to a `Data<F, R>`
    // Since this uses `cleanup` it also hinges on a correct implementation of
    // `__rustc_panic_cleanup`.
    //
    // This function cannot be marked as `unsafe` because `intrinsics::r#try`
    // expects normal function pointers.
    #[inline]
    fn do_catch<F: FnOnce() -> R, R>(data: *mut u8, payload: *mut u8) {
        // SAFETY: this is the responsibilty of the caller, see above.
        //
        // When `__rustc_panic_cleaner` is correctly implemented we can rely
        // on `obj` being the correct thing to pass to `data.p` (after wrapping
        // in `ManuallyDrop`).
        unsafe {
            let data = data as *mut Data<F, R>;
            let data = &mut (*data);
            let obj = cleanup(payload);
            data.p = ManuallyDrop::new(obj);
        }
    }
}

/// Determines whether the current thread is unwinding because of panic.
#[inline]
pub fn panicking() -> bool {
    !panic_count::is_zero()
}

/// The entry point for panicking with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `panic!()` has as low an impact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[unstable(feature = "libstd_sys_internals", reason = "used by the panic! macro", issue = "none")]
#[cold]
// If panic_immediate_abort, inline the abort call,
// otherwise avoid inlining because of it is cold path.
#[cfg_attr(not(feature = "panic_immediate_abort"), track_caller)]
#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
pub fn begin_panic_fmt(msg: &fmt::Arguments<'_>) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        intrinsics::abort()
    }

    let info = PanicInfo::internal_constructor(Some(msg), Location::caller());
    begin_panic_handler(&info)
}

/// Entry point of panics from the libcore crate (`panic_impl` lang item).
#[cfg_attr(not(test), panic_handler)]
#[unwind(allowed)]
pub fn begin_panic_handler(info: &PanicInfo<'_>) -> ! {
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
            // Lazily, the first time this gets called, run the actual string formatting.
            self.string.get_or_insert_with(|| {
                let mut s = String::new();
                drop(s.write_fmt(*inner));
                s
            })
        }
    }

    unsafe impl<'a> BoxMeUp for PanicPayload<'a> {
        fn take_box(&mut self) -> *mut (dyn Any + Send) {
            // We do two allocations here, unfortunately. But (a) they're required with the current
            // scheme, and (b) we don't handle panic + OOM properly anyway (see comment in
            // begin_panic below).
            let contents = mem::take(self.fill());
            Box::into_raw(Box::new(contents))
        }

        fn get(&mut self) -> &(dyn Any + Send) {
            self.fill()
        }
    }

    struct StrPanicPayload(&'static str);

    unsafe impl BoxMeUp for StrPanicPayload {
        fn take_box(&mut self) -> *mut (dyn Any + Send) {
            Box::into_raw(Box::new(self.0))
        }

        fn get(&mut self) -> &(dyn Any + Send) {
            &self.0
        }
    }

    let loc = info.location().unwrap(); // The current implementation always returns Some
    let msg = info.message().unwrap(); // The current implementation always returns Some
    crate::sys_common::backtrace::__rust_end_short_backtrace(move || {
        if let Some(msg) = msg.as_str() {
            rust_panic_with_hook(&mut StrPanicPayload(msg), info.message(), loc);
        } else {
            rust_panic_with_hook(&mut PanicPayload::new(msg), info.message(), loc);
        }
    })
}

/// This is the entry point of panicking for the non-format-string variants of
/// panic!() and assert!(). In particular, this is the only entry point that supports
/// arbitrary payloads, not just format strings.
#[unstable(feature = "libstd_sys_internals", reason = "used by the panic! macro", issue = "none")]
#[cfg_attr(not(test), lang = "begin_panic")]
// lang item for CTFE panic support
// never inline unless panic_immediate_abort to avoid code
// bloat at the call sites as much as possible
#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cold]
#[track_caller]
pub fn begin_panic<M: Any + Send>(msg: M) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        intrinsics::abort()
    }

    let loc = Location::caller();
    return crate::sys_common::backtrace::__rust_end_short_backtrace(move || {
        rust_panic_with_hook(&mut PanicPayload::new(msg), None, loc)
    });

    struct PanicPayload<A> {
        inner: Option<A>,
    }

    impl<A: Send + 'static> PanicPayload<A> {
        fn new(inner: A) -> PanicPayload<A> {
            PanicPayload { inner: Some(inner) }
        }
    }

    unsafe impl<A: Send + 'static> BoxMeUp for PanicPayload<A> {
        fn take_box(&mut self) -> *mut (dyn Any + Send) {
            // Note that this should be the only allocation performed in this code path. Currently
            // this means that panic!() on OOM will invoke this code path, but then again we're not
            // really ready for panic on OOM anyway. If we do start doing this, then we should
            // propagate this allocation to be performed in the parent of this thread instead of the
            // thread that's panicking.
            let data = match self.inner.take() {
                Some(a) => Box::new(a) as Box<dyn Any + Send>,
                None => process::abort(),
            };
            Box::into_raw(data)
        }

        fn get(&mut self) -> &(dyn Any + Send) {
            match self.inner {
                Some(ref a) => a,
                None => process::abort(),
            }
        }
    }
}

/// Central point for dispatching panics.
///
/// Executes the primary logic for a panic, including checking for recursive
/// panics, panic hooks, and finally dispatching to the panic runtime to either
/// abort or unwind.
fn rust_panic_with_hook(
    payload: &mut dyn BoxMeUp,
    message: Option<&fmt::Arguments<'_>>,
    location: &Location<'_>,
) -> ! {
    let panics = panic_count::increase();

    // If this is the third nested call (e.g., panics == 2, this is 0-indexed),
    // the panic hook probably triggered the last panic, otherwise the
    // double-panic check would have aborted the process. In this case abort the
    // process real quickly as we don't want to try calling it again as it'll
    // probably just panic again.
    if panics > 2 {
        util::dumb_print(format_args!("thread panicked while processing panic. aborting.\n"));
        intrinsics::abort()
    }

    unsafe {
        let mut info = PanicInfo::internal_constructor(message, location);
        HOOK_LOCK.read();
        match HOOK {
            // Some platforms (like wasm) know that printing to stderr won't ever actually
            // print anything, and if that's the case we can skip the default
            // hook. Since string formatting happens lazily when calling `payload`
            // methods, this means we avoid formatting the string at all!
            // (The panic runtime might still call `payload.take_box()` though and trigger
            // formatting.)
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
        util::dumb_print(format_args!("thread panicked while panicking. aborting.\n"));
        intrinsics::abort()
    }

    rust_panic(payload)
}

/// This is the entry point for `resume_unwind`.
/// It just forwards the payload to the panic runtime.
pub fn rust_panic_without_hook(payload: Box<dyn Any + Send>) -> ! {
    panic_count::increase();

    struct RewrapBox(Box<dyn Any + Send>);

    unsafe impl BoxMeUp for RewrapBox {
        fn take_box(&mut self) -> *mut (dyn Any + Send) {
            Box::into_raw(mem::replace(&mut self.0, Box::new(())))
        }

        fn get(&mut self) -> &(dyn Any + Send) {
            &*self.0
        }
    }

    rust_panic(&mut RewrapBox(payload))
}

/// An unmangled function (through `rustc_std_internal_symbol`) on which to slap
/// yer breakpoints.
#[inline(never)]
#[cfg_attr(not(test), rustc_std_internal_symbol)]
fn rust_panic(mut msg: &mut dyn BoxMeUp) -> ! {
    let code = unsafe {
        let obj = &mut msg as *mut &mut dyn BoxMeUp;
        __rust_start_panic(obj)
    };
    rtabort!("failed to initiate panic, error {}", code)
}
