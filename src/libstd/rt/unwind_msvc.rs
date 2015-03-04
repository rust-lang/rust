// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of Rust stack unwinding
//!
//! For background on exception handling and stack unwinding please see
//! "Exception Handling in LLVM" (llvm.org/docs/ExceptionHandling.html) and
//! documents linked from it.
//! These are also good reads:
//!     http://theofilos.cs.columbia.edu/blog/2013/09/22/base_abi/
//!     http://monoinfinito.wordpress.com/series/exception-handling-in-c/
//!     http://www.airs.com/blog/index.php?s=exception+frames
//!
//! ## A brief summary
//!
//! Exception handling happens in two phases: a search phase and a cleanup phase.
//!
//! In both phases the unwinder walks stack frames from top to bottom using
//! information from the stack frame unwind sections of the current process's
//! modules ("module" here refers to an OS module, i.e. an executable or a
//! dynamic library).
//!
//! For each stack frame, it invokes the associated "personality routine", whose
//! address is also stored in the unwind info section.
//!
//! In the search phase, the job of a personality routine is to examine exception
//! object being thrown, and to decide whether it should be caught at that stack
//! frame.  Once the handler frame has been identified, cleanup phase begins.
//!
//! In the cleanup phase, personality routines invoke cleanup code associated
//! with their stack frames (i.e. destructors).  Once stack has been unwound down
//! to the handler frame level, unwinding stops and the last personality routine
//! transfers control to its catch block.
//!
//! ## Frame unwind info registration
//!
//! Each module has its own frame unwind info section (usually ".eh_frame"), and
//! unwinder needs to know about all of them in order for unwinding to be able to
//! cross module boundaries.
//!
//! On some platforms, like Linux, this is achieved by dynamically enumerating
//! currently loaded modules via the dl_iterate_phdr() API and finding all
//! .eh_frame sections.
//!
//! Others, like Windows, require modules to actively register their unwind info
//! sections by calling __register_frame_info() API at startup.  In the latter
//! case it is essential that there is only one copy of the unwinder runtime in
//! the process.  This is usually achieved by linking to the dynamic version of
//! the unwind runtime.
//!
//! Currently Rust uses unwind runtime provided by libgcc.

use prelude::v1::*;

use any::Any;
use cell::Cell;
use cmp;
use panicking;
use fmt;
use intrinsics;
use mem;
use sync::atomic::{self, Ordering};
use sync::{Once, ONCE_INIT};

pub type Callback = fn(msg: &(Any + Send), file: &'static str, line: uint);

// Variables used for invoking callbacks when a thread starts to unwind.
//
// For more information, see below.
const MAX_CALLBACKS: uint = 16;
static CALLBACKS: [atomic::AtomicUsize; MAX_CALLBACKS] =
        [atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT,
         atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT,
         atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT,
         atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT,
         atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT,
         atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT,
         atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT,
         atomic::ATOMIC_USIZE_INIT, atomic::ATOMIC_USIZE_INIT];
static CALLBACK_CNT: atomic::AtomicUsize = atomic::ATOMIC_USIZE_INIT;

thread_local! { static PANICKING: Cell<bool> = Cell::new(false) }

/// Invoke a closure, capturing the cause of panic if one occurs.
///
/// This function will return `Ok(())` if the closure did not panic, and will
/// return `Err(cause)` if the closure panics. The `cause` returned is the
/// object with which panic was originally invoked.
///
/// This function also is unsafe for a variety of reasons:
///
/// * This is not safe to call in a nested fashion. The unwinding
///   interface for Rust is designed to have at most one try/catch block per
///   thread, not multiple. No runtime checking is currently performed to uphold
///   this invariant, so this function is not safe. A nested try/catch block
///   may result in corruption of the outer try/catch block's state, especially
///   if this is used within a thread itself.
///
/// * It is not sound to trigger unwinding while already unwinding. Rust threads
///   have runtime checks in place to ensure this invariant, but it is not
///   guaranteed that a rust thread is in place when invoking this function.
///   Unwinding twice can lead to resource leaks where some destructors are not
///   run.
pub unsafe fn try<F: FnOnce()>(f: F) -> Result<(), Box<Any + Send>> {
    f();
    Ok(())
}

/// Determines whether the current thread is unwinding because of panic.
pub fn panicking() -> bool {
    PANICKING.with(|s| s.get())
}

// An uninlined, unmangled function upon which to slap yer breakpoints
#[inline(never)]
#[no_mangle]
#[allow(private_no_mangle_fns)]
fn rust_panic(_cause: Box<Any + Send + 'static>) -> ! {
    loop {}
}

// See also: rt/rust_try.ll
#[cfg(all(not(test)))]
#[doc(hidden)]
#[allow(non_camel_case_types, non_snake_case)]
pub mod eabi {
    pub use self::EXCEPTION_DISPOSITION::*;
    use libc::c_void;

    #[repr(C)]
    pub struct EXCEPTION_RECORD;
    #[repr(C)]
    pub struct CONTEXT;
    #[repr(C)]
    pub struct DISPATCHER_CONTEXT;

    #[repr(C)]
    #[derive(Copy)]
    pub enum EXCEPTION_DISPOSITION {
        ExceptionContinueExecution,
        ExceptionContinueSearch,
        ExceptionNestedException,
        ExceptionCollidedUnwind
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    #[allow(private_no_mangle_fns)]
    extern "C" fn rust_eh_personality(
        _exceptionRecord: *mut EXCEPTION_RECORD,
        _establisherFrame: *mut c_void,
        _contextRecord: *mut CONTEXT,
        _dispatcherContext: *mut DISPATCHER_CONTEXT
    ) -> EXCEPTION_DISPOSITION
    {
        EXCEPTION_DISPOSITION::ExceptionContinueSearch
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        _exceptionRecord: *mut EXCEPTION_RECORD,
        _establisherFrame: *mut c_void,
        _contextRecord: *mut CONTEXT,
        _dispatcherContext: *mut DISPATCHER_CONTEXT
    ) -> EXCEPTION_DISPOSITION
    {
        EXCEPTION_DISPOSITION::ExceptionContinueSearch
    }
}

#[cfg(not(test))]
/// Entry point of panic from the libcore crate.
#[lang = "panic_fmt"]
pub extern fn rust_begin_unwind(msg: fmt::Arguments,
                                file: &'static str, line: uint) -> ! {
    begin_unwind_fmt(msg, &(file, line))
}

/// The entry point for unwinding with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `panic!()` has as low an impact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[inline(never)] #[cold]
#[stable(since = "1.0.0", feature = "rust1")]
pub fn begin_unwind_fmt(msg: fmt::Arguments, file_line: &(&'static str, uint)) -> ! {
    use fmt::Write;

    // We do two allocations here, unfortunately. But (a) they're
    // required with the current scheme, and (b) we don't handle
    // panic + OOM properly anyway (see comment in begin_unwind
    // below).

    let mut s = String::new();
    let _ = write!(&mut s, "{}", msg);
    begin_unwind_inner(box s, file_line)
}

/// This is the entry point of unwinding for panic!() and assert!().
#[inline(never)] #[cold] // avoid code bloat at the call sites as much as possible
#[stable(since = "1.0.0", feature = "rust1")]
pub fn begin_unwind<M: Any + Send>(msg: M, file_line: &(&'static str, uint)) -> ! {
    // Note that this should be the only allocation performed in this code path.
    // Currently this means that panic!() on OOM will invoke this code path,
    // but then again we're not really ready for panic on OOM anyway. If
    // we do start doing this, then we should propagate this allocation to
    // be performed in the parent of this thread instead of the thread that's
    // panicking.

    // see below for why we do the `Any` coercion here.
    begin_unwind_inner(box msg, file_line)
}

/// The core of the unwinding.
///
/// This is non-generic to avoid instantiation bloat in other crates
/// (which makes compilation of small crates noticeably slower). (Note:
/// we need the `Any` object anyway, we're not just creating it to
/// avoid being generic.)
///
/// Doing this split took the LLVM IR line counts of `fn main() { panic!()
/// }` from ~1900/3700 (-O/no opts) to 180/590.
#[inline(never)] #[cold] // this is the slow path, please never inline this
fn begin_unwind_inner(msg: Box<Any + Send>, file_line: &(&'static str, uint)) -> ! {
    // Make sure the default panic handler is registered before we look at the
    // callbacks.
    static INIT: Once = ONCE_INIT;
    INIT.call_once(|| unsafe { register(panicking::on_panic); });

    // First, invoke call the user-defined callbacks triggered on thread panic.
    //
    // By the time that we see a callback has been registered (by reading
    // MAX_CALLBACKS), the actual callback itself may have not been stored yet,
    // so we just chalk it up to a race condition and move on to the next
    // callback. Additionally, CALLBACK_CNT may briefly be higher than
    // MAX_CALLBACKS, so we're sure to clamp it as necessary.
    let callbacks = {
        let amt = CALLBACK_CNT.load(Ordering::SeqCst);
        &CALLBACKS[..cmp::min(amt, MAX_CALLBACKS)]
    };
    for cb in callbacks {
        match cb.load(Ordering::SeqCst) {
            0 => {}
            n => {
                let f: Callback = unsafe { mem::transmute(n) };
                let (file, line) = *file_line;
                f(&*msg, file, line);
            }
        }
    };

    // Now that we've run all the necessary unwind callbacks, we actually
    // perform the unwinding.
    if panicking() {
        // If a thread panics while it's already unwinding then we
        // have limited options. Currently our preference is to
        // just abort. In the future we may consider resuming
        // unwinding or otherwise exiting the thread cleanly.
        rterrln!("thread panicked while panicking. aborting.");
        unsafe { intrinsics::abort() }
    }
    PANICKING.with(|s| s.set(true));
    rust_panic(msg);
}

/// Register a callback to be invoked when a thread unwinds.
///
/// This is an unsafe and experimental API which allows for an arbitrary
/// callback to be invoked when a thread panics. This callback is invoked on both
/// the initial unwinding and a double unwinding if one occurs. Additionally,
/// the local `Task` will be in place for the duration of the callback, and
/// the callback must ensure that it remains in place once the callback returns.
///
/// Only a limited number of callbacks can be registered, and this function
/// returns whether the callback was successfully registered or not. It is not
/// currently possible to unregister a callback once it has been registered.
#[unstable(feature = "std_misc")]
pub unsafe fn register(f: Callback) -> bool {
    match CALLBACK_CNT.fetch_add(1, Ordering::SeqCst) {
        // The invocation code has knowledge of this window where the count has
        // been incremented, but the callback has not been stored. We're
        // guaranteed that the slot we're storing into is 0.
        n if n < MAX_CALLBACKS => {
            let prev = CALLBACKS[n].swap(mem::transmute(f), Ordering::SeqCst);
            rtassert!(prev == 0);
            true
        }
        // If we accidentally bumped the count too high, pull it back.
        _ => {
            CALLBACK_CNT.store(MAX_CALLBACKS, Ordering::SeqCst);
            false
        }
    }
}
