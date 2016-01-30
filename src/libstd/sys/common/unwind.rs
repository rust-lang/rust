// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Common unwinding support to all platforms.
//!
//! This module does not contain the lowest-level implementation of unwinding
//! itself, but rather contains some of the Rust APIs used as entry points to
//! unwinding itself. For example this is the location of the lang item
//! `panic_fmt` which is the entry point of all panics from libcore.
//!
//! For details on how each platform implements unwinding, see the
//! platform-specific `sys::unwind` module.

use prelude::v1::*;

use any::Any;
use panicking::{self,PANIC_COUNT};
use fmt;
use intrinsics;
use sys::unwind;

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
    let mut f = Some(f);
    return inner_try(try_fn::<F>, &mut f as *mut _ as *mut u8);

    fn try_fn<F: FnOnce()>(opt_closure: *mut u8) {
        let opt_closure = opt_closure as *mut Option<F>;
        unsafe { (*opt_closure).take().unwrap()(); }
    }
}

#[cfg(not(stage0))]
unsafe fn inner_try(f: fn(*mut u8), data: *mut u8)
                    -> Result<(), Box<Any + Send>> {
    PANIC_COUNT.with(|s| {
        let prev = s.get();
        s.set(0);

        // The "payload" here is a platform-specific region of memory which is
        // used to transmit information about the exception being thrown from
        // the point-of-throw back to this location.
        //
        // A pointer to this data is passed to the `try` intrinsic itself,
        // allowing this function, the `try` intrinsic, unwind::payload(), and
        // unwind::cleanup() to all work in concert to transmit this
        // information.
        //
        // More information about what this pointer actually is can be found in
        // each implementation as well as browsing the compiler source itself.
        let mut payload = unwind::payload();
        let r = intrinsics::try(f, data, &mut payload as *mut _ as *mut _);
        s.set(prev);
        if r == 0 {
            Ok(())
        } else {
            Err(unwind::cleanup(payload))
        }
    })
}

#[cfg(stage0)]
unsafe fn inner_try(f: fn(*mut u8), data: *mut u8)
                    -> Result<(), Box<Any + Send>> {
    PANIC_COUNT.with(|s| {
        let prev = s.get();
        s.set(0);
        let ep = intrinsics::try(f, data);
        s.set(prev);
        if ep.is_null() {
            Ok(())
        } else {
            Err(unwind::cleanup(ep))
        }
    })
}

/// Determines whether the current thread is unwinding because of panic.
pub fn panicking() -> bool {
    PANIC_COUNT.with(|s| s.get() != 0)
}

// An uninlined, unmangled function upon which to slap yer breakpoints
#[inline(never)]
#[no_mangle]
#[allow(private_no_mangle_fns)]
pub fn rust_panic(cause: Box<Any + Send + 'static>) -> ! {
    unsafe {
        unwind::panic(cause)
    }
}

#[cfg(not(test))]
/// Entry point of panic from the libcore crate.
#[lang = "panic_fmt"]
#[unwind]
pub extern fn rust_begin_unwind(msg: fmt::Arguments,
                                file: &'static str, line: u32) -> ! {
    begin_unwind_fmt(msg, &(file, line))
}

/// The entry point for unwinding with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `panic!()` has as low an impact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[unstable(feature = "libstd_sys_internals",
           reason = "used by the panic! macro",
           issue = "0")]
#[inline(never)] #[cold]
pub fn begin_unwind_fmt(msg: fmt::Arguments, file_line: &(&'static str, u32)) -> ! {
    use fmt::Write;

    // We do two allocations here, unfortunately. But (a) they're
    // required with the current scheme, and (b) we don't handle
    // panic + OOM properly anyway (see comment in begin_unwind
    // below).

    let mut s = String::new();
    let _ = s.write_fmt(msg);
    begin_unwind_inner(Box::new(s), file_line)
}

/// This is the entry point of unwinding for panic!() and assert!().
#[unstable(feature = "libstd_sys_internals",
           reason = "used by the panic! macro",
           issue = "0")]
#[inline(never)] #[cold] // avoid code bloat at the call sites as much as possible
pub fn begin_unwind<M: Any + Send>(msg: M, file_line: &(&'static str, u32)) -> ! {
    // Note that this should be the only allocation performed in this code path.
    // Currently this means that panic!() on OOM will invoke this code path,
    // but then again we're not really ready for panic on OOM anyway. If
    // we do start doing this, then we should propagate this allocation to
    // be performed in the parent of this thread instead of the thread that's
    // panicking.

    // see below for why we do the `Any` coercion here.
    begin_unwind_inner(Box::new(msg), file_line)
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
fn begin_unwind_inner(msg: Box<Any + Send>,
                      file_line: &(&'static str, u32)) -> ! {
    let (file, line) = *file_line;

    // First, invoke the default panic handler.
    panicking::on_panic(&*msg, file, line);

    // Finally, perform the unwinding.
    rust_panic(msg);
}
