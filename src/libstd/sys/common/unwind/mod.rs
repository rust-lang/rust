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
//!     http://mentorembedded.github.io/cxx-abi/abi-eh.html
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
//! In the cleanup phase, the unwinder invokes each personality routine again.
//! This time it decides which (if any) cleanup code needs to be run for
//! the current stack frame.  If so, the control is transferred to a special branch
//! in the function body, the "landing pad", which invokes destructors, frees memory,
//! etc.  At the end of the landing pad, control is transferred back to the unwinder
//! and unwinding resumes.
//!
//! Once stack has been unwound down to the handler frame level, unwinding stops
//! and the last personality routine transfers control to the catch block.
//!
//! ## `eh_personality` and `eh_unwind_resume`
//!
//! These language items are used by the compiler when generating unwind info.
//! The first one is the personality routine described above.  The second one
//! allows compilation target to customize the process of resuming unwind at the
//! end of the landing pads.  `eh_unwind_resume` is used only if `custom_unwind_resume`
//! flag in the target options is set.
//!
//! ## Frame unwind info registration
//!
//! Each module's image contains a frame unwind info section (usually ".eh_frame").
//! When a module is loaded/unloaded into the process, the unwinder must be informed
//! about the location of this section in memory. The methods of achieving that vary
//! by the platform.
//! On some (e.g. Linux), the unwinder can discover unwind info sections on its own
//! (by dynamically enumerating currently loaded modules via the dl_iterate_phdr() API
//! and finding their ".eh_frame" sections);
//! Others, like Windows, require modules to actively register their unwind info
//! sections via unwinder API (see `rust_eh_register_frames`/`rust_eh_unregister_frames`).

#![allow(dead_code)]
#![allow(unused_imports)]

use prelude::v1::*;

use any::Any;
use boxed;
use cmp;
use panicking::{self,PANIC_COUNT};
use fmt;
use intrinsics;
use mem;
use sync::atomic::{self, Ordering};
use sys_common::mutex::Mutex;

// The actual unwinding implementation is cfg'd here, and we've got two current
// implementations. One goes through SEH on Windows and the other goes through
// libgcc via the libunwind-like API.

// i686-pc-windows-msvc
#[cfg(all(windows, target_arch = "x86", target_env = "msvc"))]
#[path = "seh.rs"] #[doc(hidden)]
pub mod imp;

// x86_64-pc-windows-*
#[cfg(all(windows, target_arch = "x86_64"))]
#[path = "seh64_gnu.rs"] #[doc(hidden)]
pub mod imp;

// i686-pc-windows-gnu and all others
#[cfg(any(unix, all(windows, target_arch = "x86", target_env = "gnu")))]
#[path = "gcc.rs"] #[doc(hidden)]
pub mod imp;

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

    // If an inner function were not used here, then this generic function `try`
    // uses the native symbol `rust_try`, for which the code is statically
    // linked into the standard library. This means that the DLL for the
    // standard library must have `rust_try` as an exposed symbol that
    // downstream crates can link against (because monomorphizations of `try` in
    // downstream crates will have a reference to the `rust_try` symbol).
    //
    // On MSVC this requires the symbol `rust_try` to be tagged with
    // `dllexport`, but it's easier to not have conditional `src/rt/rust_try.ll`
    // files and instead just have this non-generic shim the compiler can take
    // care of exposing correctly.
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
                Err(imp::cleanup(ep))
            }
        })
    }

    fn try_fn<F: FnOnce()>(opt_closure: *mut u8) {
        let opt_closure = opt_closure as *mut Option<F>;
        unsafe { (*opt_closure).take().unwrap()(); }
    }

    extern {
        // Rust's try-catch
        // When f(...) returns normally, the return value is null.
        // When f(...) throws, the return value is a pointer to the caught
        // exception object.
        fn rust_try(f: extern fn(*mut u8),
                    data: *mut u8) -> *mut u8;
    }
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
        imp::panic(cause)
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
