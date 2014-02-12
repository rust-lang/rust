// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
use c_str::CString;
use cast;
use fmt;
use kinds::Send;
use mem;
use option::{Some, None, Option};
use prelude::drop;
use ptr::RawPtr;
use result::{Err, Ok};
use rt::local::Local;
use rt::task::Task;
use str::Str;
use task::TaskResult;
use unstable::intrinsics;

use uw = self::libunwind;

#[allow(dead_code)]
mod libunwind {
    //! Unwind library interface

    #[allow(non_camel_case_types)];
    #[allow(dead_code)]; // these are just bindings

    use libc::{uintptr_t};

    #[cfg(not(target_arch = "arm"))]
    #[repr(C)]
    pub enum _Unwind_Action
    {
        _UA_SEARCH_PHASE = 1,
        _UA_CLEANUP_PHASE = 2,
        _UA_HANDLER_FRAME = 4,
        _UA_FORCE_UNWIND = 8,
        _UA_END_OF_STACK = 16,
    }

    #[cfg(target_arch = "arm")]
    #[repr(C)]
    pub enum _Unwind_State
    {
      _US_VIRTUAL_UNWIND_FRAME = 0,
      _US_UNWIND_FRAME_STARTING = 1,
      _US_UNWIND_FRAME_RESUME = 2,
      _US_ACTION_MASK = 3,
      _US_FORCE_UNWIND = 8,
      _US_END_OF_STACK = 16
    }

    #[repr(C)]
    pub enum _Unwind_Reason_Code {
        _URC_NO_REASON = 0,
        _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
        _URC_FATAL_PHASE2_ERROR = 2,
        _URC_FATAL_PHASE1_ERROR = 3,
        _URC_NORMAL_STOP = 4,
        _URC_END_OF_STACK = 5,
        _URC_HANDLER_FOUND = 6,
        _URC_INSTALL_CONTEXT = 7,
        _URC_CONTINUE_UNWIND = 8,
        _URC_FAILURE = 9, // used only by ARM EABI
    }

    pub type _Unwind_Exception_Class = u64;

    pub type _Unwind_Word = uintptr_t;

    #[cfg(target_arch = "x86")]
    pub static unwinder_private_data_size: int = 5;

    #[cfg(target_arch = "x86_64")]
    pub static unwinder_private_data_size: int = 2;

    #[cfg(target_arch = "arm")]
    pub static unwinder_private_data_size: int = 20;

    pub struct _Unwind_Exception {
        exception_class: _Unwind_Exception_Class,
        exception_cleanup: _Unwind_Exception_Cleanup_Fn,
        private: [_Unwind_Word, ..unwinder_private_data_size],
    }

    pub enum _Unwind_Context {}

    pub type _Unwind_Exception_Cleanup_Fn = extern "C" fn(unwind_code: _Unwind_Reason_Code,
                                                          exception: *_Unwind_Exception);

    extern "C" {
        pub fn _Unwind_RaiseException(exception: *_Unwind_Exception) -> _Unwind_Reason_Code;
        pub fn _Unwind_DeleteException(exception: *_Unwind_Exception);
    }
}

pub struct Unwinder {
    priv unwinding: bool,
    priv cause: Option<~Any>
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
        use unstable::raw::Closure;
        use libc::{c_void};

        unsafe {
            let closure: Closure = cast::transmute(f);
            let ep = rust_try(try_fn, closure.code as *c_void,
                              closure.env as *c_void);
            if !ep.is_null() {
                rtdebug!("caught {}", (*ep).exception_class);
                uw::_Unwind_DeleteException(ep);
            }
        }

        extern fn try_fn(code: *c_void, env: *c_void) {
            unsafe {
                let closure: || = cast::transmute(Closure {
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

    pub fn begin_unwind(&mut self, cause: ~Any) -> ! {
        rtdebug!("begin_unwind()");

        self.unwinding = true;
        self.cause = Some(cause);

        rust_fail();

        // An uninlined, unmangled function upon which to slap yer breakpoints
        #[inline(never)]
        #[no_mangle]
        fn rust_fail() -> ! {
            unsafe {
                let exception = ~uw::_Unwind_Exception {
                    exception_class: rust_exception_class(),
                    exception_cleanup: exception_cleanup,
                    private: [0, ..uw::unwinder_private_data_size],
                };
                let error = uw::_Unwind_RaiseException(cast::transmute(exception));
                rtabort!("Could not unwind stack, error = {}", error as int)
            }

            extern "C" fn exception_cleanup(_unwind_code: uw::_Unwind_Reason_Code,
                                            exception: *uw::_Unwind_Exception) {
                rtdebug!("exception_cleanup()");
                unsafe {
                    let _: ~uw::_Unwind_Exception = cast::transmute(exception);
                }
            }
        }
    }

    pub fn result(&mut self) -> TaskResult {
        if self.unwinding {
            Err(self.cause.take().unwrap())
        } else {
            Ok(())
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
pub mod eabi {
    use uw = super::libunwind;
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
    #[no_mangle] // so we can reference it by name from middle/trans/base.rs
    pub extern "C" fn rust_eh_personality(
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
pub mod eabi {
    use uw = super::libunwind;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(state: uw::_Unwind_State,
                                ue_header: *uw::_Unwind_Exception,
                                context: *uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    #[no_mangle] // so we can reference it by name from middle/trans/base.rs
    pub extern "C" fn rust_eh_personality(
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

/// This is the entry point of unwinding for things like lang items and such.
/// The arguments are normally generated by the compiler, and need to
/// have static lifetimes.
#[inline(never)] #[cold] // this is the slow path, please never inline this
pub fn begin_unwind_raw(msg: *u8, file: *u8, line: uint) -> ! {
    use libc::c_char;
    #[inline]
    fn static_char_ptr(p: *u8) -> &'static str {
        let s = unsafe { CString::new(p as *c_char, false) };
        match s.as_str() {
            Some(s) => unsafe { cast::transmute::<&str, &'static str>(s) },
            None => rtabort!("message wasn't utf8?")
        }
    }

    let msg = static_char_ptr(msg);
    let file = static_char_ptr(file);

    begin_unwind(msg, file, line as uint)
}

/// The entry point for unwinding with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `fail!()` has as low an implact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[inline(never)] #[cold]
pub fn begin_unwind_fmt(msg: &fmt::Arguments, file: &'static str, line: uint) -> ! {
    // We do two allocations here, unfortunately. But (a) they're
    // required with the current scheme, and (b) we don't handle
    // failure + OOM properly anyway (see comment in begin_unwind
    // below).
    begin_unwind_inner(~fmt::format(msg), file, line)
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
    begin_unwind_inner(~msg, file, line)
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
fn begin_unwind_inner(msg: ~Any, file: &'static str, line: uint) -> ! {
    let mut task;
    {
        let msg_s = match msg.as_ref::<&'static str>() {
            Some(s) => *s,
            None => match msg.as_ref::<~str>() {
                Some(s) => s.as_slice(),
                None => "~Any",
            }
        };

        // It is assumed that all reasonable rust code will have a local task at
        // all times. This means that this `try_take` will succeed almost all of
        // the time. There are border cases, however, when the runtime has
        // *almost* set up the local task, but hasn't quite gotten there yet. In
        // order to get some better diagnostics, we print on failure and
        // immediately abort the whole process if there is no local task
        // available.
        let opt_task: Option<~Task> = Local::try_take();
        task = match opt_task {
            Some(t) => t,
            None => {
                rterrln!("failed at '{}', {}:{}", msg_s, file, line);
                unsafe { intrinsics::abort() }
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
                    let _err = format_args!(|args| ::fmt::writeln(stderr, args),
                                            "task '{}' failed at '{}', {}:{}",
                                            n, msg_s, file, line);
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
                    rterrln!("task '{}' failed at '{}', {}:{}", n, msg_s,
                             file, line);
                }
            }
        }
        task.name = name;

        if task.unwinder.unwinding {
            // If a task fails while it's already unwinding then we
            // have limited options. Currently our preference is to
            // just abort. In the future we may consider resuming
            // unwinding or otherwise exiting the task cleanly.
            rterrln!("task failed during unwinding (double-failure - total drag!)")
            rterrln!("rust must abort now. so sorry.");
            unsafe { intrinsics::abort() }
        }
    }

    // The unwinder won't actually use the task at all, so we put the task back
    // into TLS right before we invoke the unwinder, but this means we need an
    // unsafe reference back to the unwinder once it's in TLS.
    Local::put(task);
    unsafe {
        let task: *mut Task = Local::unsafe_borrow();
        (*task).unwinder.begin_unwind(msg);
    }
}
