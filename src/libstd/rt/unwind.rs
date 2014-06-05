// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Stack unwinding

// Implementation of Rust stack unwinding
//
// For background on exception handling and stack unwinding please see
// "Exception Handling in LLVM" (llvm.org/docs/ExceptionHandling.html) and
// documents linked from it.
// These are also good reads:
//     http://theofilos.cs.columbia.edu/blog/2013/09/22/base_abi/
//     http://monoinfinito.wordpress.com/series/exception-handling-in-c/
//     http://www.airs.com/blog/index.php?s=exception+frames
//
// ~~~ A brief summary ~~~
// Exception handling happens in two phases: a search phase and a cleanup phase.
//
// In both phases the unwinder walks stack frames from top to bottom using
// information from the stack frame unwind sections of the current process's
// modules ("module" here refers to an OS module, i.e. an executable or a
// dynamic library).
//
// For each stack frame, it invokes the associated "personality routine", whose
// address is also stored in the unwind info section.
//
// In the search phase, the job of a personality routine is to examine exception
// object being thrown, and to decide whether it should be caught at that stack
// frame.  Once the handler frame has been identified, cleanup phase begins.
//
// In the cleanup phase, personality routines invoke cleanup code associated
// with their stack frames (i.e. destructors).  Once stack has been unwound down
// to the handler frame level, unwinding stops and the last personality routine
// transfers control to its' catch block.
//
// ~~~ Frame unwind info registration ~~~
// Each module has its' own frame unwind info section (usually ".eh_frame"), and
// unwinder needs to know about all of them in order for unwinding to be able to
// cross module boundaries.
//
// On some platforms, like Linux, this is achieved by dynamically enumerating
// currently loaded modules via the dl_iterate_phdr() API and finding all
// .eh_frame sections.
//
// Others, like Windows, require modules to actively register their unwind info
// sections by calling __register_frame_info() API at startup.  In the latter
// case it is essential that there is only one copy of the unwinder runtime in
// the process.  This is usually achieved by linking to the dynamic version of
// the unwind runtime.
//
// Currently Rust uses unwind runtime provided by libgcc.

use any::{Any, AnyRefExt};
use fmt;
use intrinsics;
use kinds::Send;
use mem;
use option::{Some, None, Option};
use owned::Box;
use prelude::drop;
use ptr::RawPtr;
use result::{Err, Ok, Result};
use rt::backtrace;
use rt::local::Local;
use rt::task::Task;
use str::Str;
use string::String;
use task::TaskResult;

use uw = rt::libunwind;

pub struct Unwinder {
    unwinding: bool,
    cause: Option<Box<Any:Send>>
}

struct Exception {
    uwe: uw::_Unwind_Exception,
    cause: Option<Box<Any:Send>>,
}

impl Unwinder {
    pub fn new() -> Unwinder {
        Unwinder {
            unwinding: false,
            cause: None,
        }
    }

    pub fn unwinding(&self) -> bool {
        self.unwinding
    }

    pub fn try(&mut self, f: ||) {
        self.cause = unsafe { try(f) }.err();
    }

    pub fn result(&mut self) -> TaskResult {
        if self.unwinding {
            Err(self.cause.take().unwrap())
        } else {
            Ok(())
        }
    }
}

/// Invoke a closure, capturing the cause of failure if one occurs.
///
/// This function will return `None` if the closure did not fail, and will
/// return `Some(cause)` if the closure fails. The `cause` returned is the
/// object with which failure was originally invoked.
///
/// This function also is unsafe for a variety of reasons:
///
/// * This is not safe to call in a nested fashion. The unwinding
///   interface for Rust is designed to have at most one try/catch block per
///   task, not multiple. No runtime checking is currently performed to uphold
///   this invariant, so this function is not safe. A nested try/catch block
///   may result in corruption of the outer try/catch block's state, especially
///   if this is used within a task itself.
///
/// * It is not sound to trigger unwinding while already unwinding. Rust tasks
///   have runtime checks in place to ensure this invariant, but it is not
///   guaranteed that a rust task is in place when invoking this function.
///   Unwinding twice can lead to resource leaks where some destructors are not
///   run.
pub unsafe fn try(f: ||) -> Result<(), Box<Any:Send>> {
    use raw::Closure;
    use libc::{c_void};

    let closure: Closure = mem::transmute(f);
    let ep = rust_try(try_fn, closure.code as *c_void,
                      closure.env as *c_void);
    return if ep.is_null() {
        Ok(())
    } else {
        let my_ep = ep as *mut Exception;
        rtdebug!("caught {}", (*my_ep).uwe.exception_class);
        let cause = (*my_ep).cause.take();
        uw::_Unwind_DeleteException(ep);
        Err(cause.unwrap())
    };

    extern fn try_fn(code: *c_void, env: *c_void) {
        unsafe {
            let closure: || = mem::transmute(Closure {
                code: code as *(),
                env: env as *(),
            });
            closure();
        }
    }

    extern {
        // Rust's try-catch
        // When f(...) returns normally, the return value is null.
        // When f(...) throws, the return value is a pointer to the caught
        // exception object.
        fn rust_try(f: extern "C" fn(*c_void, *c_void),
                    code: *c_void,
                    data: *c_void) -> *uw::_Unwind_Exception;
    }
}

// An uninlined, unmangled function upon which to slap yer breakpoints
#[inline(never)]
#[no_mangle]
fn rust_fail(cause: Box<Any:Send>) -> ! {
    rtdebug!("begin_unwind()");

    unsafe {
        let exception = box Exception {
            uwe: uw::_Unwind_Exception {
                exception_class: rust_exception_class(),
                exception_cleanup: exception_cleanup,
                private: [0, ..uw::unwinder_private_data_size],
            },
            cause: Some(cause),
        };
        let error = uw::_Unwind_RaiseException(mem::transmute(exception));
        rtabort!("Could not unwind stack, error = {}", error as int)
    }

    extern fn exception_cleanup(_unwind_code: uw::_Unwind_Reason_Code,
                                exception: *uw::_Unwind_Exception) {
        rtdebug!("exception_cleanup()");
        unsafe {
            let _: Box<Exception> = mem::transmute(exception);
        }
    }
}

// Rust's exception class identifier.  This is used by personality routines to
// determine whether the exception was thrown by their own runtime.
fn rust_exception_class() -> uw::_Unwind_Exception_Class {
    // M O Z \0  R U S T -- vendor, language
    0x4d4f5a_00_52555354
}

// We could implement our personality routine in pure Rust, however exception
// info decoding is tedious.  More importantly, personality routines have to
// handle various platform quirks, which are not fun to maintain.  For this
// reason, we attempt to reuse personality routine of the C language:
// __gcc_personality_v0.
//
// Since C does not support exception catching, __gcc_personality_v0 simply
// always returns _URC_CONTINUE_UNWIND in search phase, and always returns
// _URC_INSTALL_CONTEXT (i.e. "invoke cleanup code") in cleanup phase.
//
// This is pretty close to Rust's exception handling approach, except that Rust
// does have a single "catch-all" handler at the bottom of each task's stack.
// So we have two versions:
// - rust_eh_personality, used by all cleanup landing pads, which never catches,
//   so the behavior of __gcc_personality_v0 is perfectly adequate there, and
// - rust_eh_personality_catch, used only by rust_try(), which always catches.
//   This is achieved by overriding the return value in search phase to always
//   say "catch!".

#[cfg(not(target_arch = "arm"), not(test))]
#[doc(hidden)]
#[allow(visible_private_types)]
pub mod eabi {
    use uw = rt::libunwind;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *uw::_Unwind_Exception,
                                context: *uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    extern fn eh_personality(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *uw::_Unwind_Exception,
        context: *uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_v0(version, actions, exception_class, ue_header,
                                 context)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *uw::_Unwind_Exception,
        context: *uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            unsafe {
                 __gcc_personality_v0(version, actions, exception_class, ue_header,
                                      context)
            }
        }
    }
}

// ARM EHABI uses a slightly different personality routine signature,
// but otherwise works the same.
#[cfg(target_arch = "arm", not(test))]
#[allow(visible_private_types)]
pub mod eabi {
    use uw = rt::libunwind;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(state: uw::_Unwind_State,
                                ue_header: *uw::_Unwind_Exception,
                                context: *uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    extern "C" fn eh_personality(
        state: uw::_Unwind_State,
        ue_header: *uw::_Unwind_Exception,
        context: *uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_v0(state, ue_header, context)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        state: uw::_Unwind_State,
        ue_header: *uw::_Unwind_Exception,
        context: *uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (state as c_int & uw::_US_ACTION_MASK as c_int)
                           == uw::_US_VIRTUAL_UNWIND_FRAME as c_int { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            unsafe {
                 __gcc_personality_v0(state, ue_header, context)
            }
        }
    }
}

// Entry point of failure from the libcore crate
#[cfg(not(test))]
#[lang = "begin_unwind"]
pub extern fn rust_begin_unwind(msg: &fmt::Arguments,
                                file: &'static str, line: uint) -> ! {
    begin_unwind_fmt(msg, file, line)
}

/// The entry point for unwinding with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `fail!()` has as low an impact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[inline(never)] #[cold]
pub fn begin_unwind_fmt(msg: &fmt::Arguments, file: &'static str,
                        line: uint) -> ! {
    // We do two allocations here, unfortunately. But (a) they're
    // required with the current scheme, and (b) we don't handle
    // failure + OOM properly anyway (see comment in begin_unwind
    // below).
    begin_unwind_inner(box fmt::format(msg), file, line)
}

/// This is the entry point of unwinding for fail!() and assert!().
#[inline(never)] #[cold] // avoid code bloat at the call sites as much as possible
pub fn begin_unwind<M: Any + Send>(msg: M, file: &'static str, line: uint) -> ! {
    // Note that this should be the only allocation performed in this code path.
    // Currently this means that fail!() on OOM will invoke this code path,
    // but then again we're not really ready for failing on OOM anyway. If
    // we do start doing this, then we should propagate this allocation to
    // be performed in the parent of this task instead of the task that's
    // failing.

    // see below for why we do the `Any` coercion here.
    begin_unwind_inner(box msg, file, line)
}


/// The core of the unwinding.
///
/// This is non-generic to avoid instantiation bloat in other crates
/// (which makes compilation of small crates noticably slower). (Note:
/// we need the `Any` object anyway, we're not just creating it to
/// avoid being generic.)
///
/// Do this split took the LLVM IR line counts of `fn main() { fail!()
/// }` from ~1900/3700 (-O/no opts) to 180/590.
#[inline(never)] #[cold] // this is the slow path, please never inline this
fn begin_unwind_inner(msg: Box<Any:Send>,
                      file: &'static str,
                      line: uint) -> ! {
    // First up, print the message that we're failing
    print_failure(msg, file, line);

    let opt_task: Option<Box<Task>> = Local::try_take();
    match opt_task {
        Some(mut task) => {
            // Now that we've printed why we're failing, do a check
            // to make sure that we're not double failing.
            //
            // If a task fails while it's already unwinding then we
            // have limited options. Currently our preference is to
            // just abort. In the future we may consider resuming
            // unwinding or otherwise exiting the task cleanly.
            if task.unwinder.unwinding {
                rterrln!("task failed during unwinding. aborting.");

                // Don't print the backtrace twice (it would have already been
                // printed if logging was enabled).
                if !backtrace::log_enabled() {
                    let mut err = ::rt::util::Stderr;
                    let _err = backtrace::write(&mut err);
                }
                unsafe { intrinsics::abort() }
            }

            // Finally, we've printed our failure and figured out we're not in a
            // double failure, so flag that we've started to unwind and then
            // actually unwind.  Be sure that the task is in TLS so destructors
            // can do fun things like I/O.
            task.unwinder.unwinding = true;
            Local::put(task);
        }
        None => {}
    }
    rust_fail(msg)
}

/// Given a failure message and the location that it occurred, prints the
/// message to the local task's appropriate stream.
///
/// This function currently handles three cases:
///
///     - There is no local task available. In this case the error is printed to
///       stderr.
///     - There is a local task available, but it does not have a stderr handle.
///       In this case the message is also printed to stderr.
///     - There is a local task available, and it has a stderr handle. The
///       message is printed to the handle given in this case.
fn print_failure(msg: &Any:Send, file: &str, line: uint) {
    let msg = match msg.as_ref::<&'static str>() {
        Some(s) => *s,
        None => match msg.as_ref::<String>() {
            Some(s) => s.as_slice(),
            None => "Box<Any>",
        }
    };

    // It is assumed that all reasonable rust code will have a local task at
    // all times. This means that this `try_take` will succeed almost all of
    // the time. There are border cases, however, when the runtime has
    // *almost* set up the local task, but hasn't quite gotten there yet. In
    // order to get some better diagnostics, we print on failure and
    // immediately abort the whole process if there is no local task
    // available.
    let mut task: Box<Task> = match Local::try_take() {
        Some(t) => t,
        None => {
            rterrln!("failed at '{}', {}:{}", msg, file, line);
            if backtrace::log_enabled() {
                let mut err = ::rt::util::Stderr;
                let _err = backtrace::write(&mut err);
            } else {
                rterrln!("run with `RUST_BACKTRACE=1` to see a backtrace");
            }
            return
        }
    };

    // See comments in io::stdio::with_task_stdout as to why we have to be
    // careful when using an arbitrary I/O handle from the task. We
    // essentially need to dance to make sure when a task is in TLS when
    // running user code.
    let name = task.name.take();
    {
        let n = name.as_ref().map(|n| n.as_slice()).unwrap_or("<unnamed>");

        match task.stderr.take() {
            Some(mut stderr) => {
                Local::put(task);
                // FIXME: what to do when the task printing fails?
                let _err = write!(stderr,
                                  "task '{}' failed at '{}', {}:{}\n",
                                  n, msg, file, line);
                if backtrace::log_enabled() {
                    let _err = backtrace::write(stderr);
                }
                task = Local::take();

                match mem::replace(&mut task.stderr, Some(stderr)) {
                    Some(prev) => {
                        Local::put(task);
                        drop(prev);
                        task = Local::take();
                    }
                    None => {}
                }
            }
            None => {
                rterrln!("task '{}' failed at '{}', {}:{}", n, msg, file, line);
                if backtrace::log_enabled() {
                    let mut err = ::rt::util::Stderr;
                    let _err = backtrace::write(&mut err);
                }
            }
        }
    }
    task.name = name;
    Local::put(task);
}
