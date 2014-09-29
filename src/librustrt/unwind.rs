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

use core::prelude::*;

use alloc::boxed::Box;
use collections::string::String;
use collections::str::StrAllocating;
use collections::vec::Vec;
use core::any::Any;
use core::atomic;
use core::cmp;
use core::fmt;
use core::intrinsics;
use core::mem;
use core::raw::Closure;
use libc::c_void;

use local::Local;
use task::Task;

use libunwind as uw;

pub struct Unwinder {
    unwinding: bool,
}

struct Exception {
    uwe: uw::_Unwind_Exception,
    cause: Option<Box<Any + Send>>,
}

pub type Callback = fn(msg: &Any + Send, file: &'static str, line: uint);

// Variables used for invoking callbacks when a task starts to unwind.
//
// For more information, see below.
static MAX_CALLBACKS: uint = 16;
static mut CALLBACKS: [atomic::AtomicUint, ..MAX_CALLBACKS] =
        [atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT,
         atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT,
         atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT,
         atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT,
         atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT,
         atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT,
         atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT,
         atomic::INIT_ATOMIC_UINT, atomic::INIT_ATOMIC_UINT];
static mut CALLBACK_CNT: atomic::AtomicUint = atomic::INIT_ATOMIC_UINT;

impl Unwinder {
    pub fn new() -> Unwinder {
        Unwinder {
            unwinding: false,
        }
    }

    pub fn unwinding(&self) -> bool {
        self.unwinding
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
pub unsafe fn try(f: ||) -> ::core::result::Result<(), Box<Any + Send>> {
    let closure: Closure = mem::transmute(f);
    let ep = rust_try(try_fn, closure.code as *mut c_void,
                      closure.env as *mut c_void);
    return if ep.is_null() {
        Ok(())
    } else {
        let my_ep = ep as *mut Exception;
        rtdebug!("caught {}", (*my_ep).uwe.exception_class);
        let cause = (*my_ep).cause.take();
        uw::_Unwind_DeleteException(ep);
        Err(cause.unwrap())
    };

    extern fn try_fn(code: *mut c_void, env: *mut c_void) {
        unsafe {
            let closure: || = mem::transmute(Closure {
                code: code as *mut (),
                env: env as *mut (),
            });
            closure();
        }
    }

    #[link(name = "rustrt_native", kind = "static")]
    #[cfg(not(test))]
    extern {}

    extern {
        // Rust's try-catch
        // When f(...) returns normally, the return value is null.
        // When f(...) throws, the return value is a pointer to the caught
        // exception object.
        fn rust_try(f: extern "C" fn(*mut c_void, *mut c_void),
                    code: *mut c_void,
                    data: *mut c_void) -> *mut uw::_Unwind_Exception;
    }
}

// An uninlined, unmangled function upon which to slap yer breakpoints
#[inline(never)]
#[no_mangle]
fn rust_fail(cause: Box<Any + Send>) -> ! {
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
                                exception: *mut uw::_Unwind_Exception) {
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
// So we have two versions of the personality routine:
// - rust_eh_personality, used by all cleanup landing pads, which never catches,
//   so the behavior of __gcc_personality_v0 is perfectly adequate there, and
// - rust_eh_personality_catch, used only by rust_try(), which always catches.
//
// Note, however, that for implementation simplicity, rust_eh_personality_catch
// lacks code to install a landing pad, so in order to obtain exception object
// pointer (which it needs to return upstream), rust_try() employs another trick:
// it calls into the nested rust_try_inner(), whose landing pad does not resume
// unwinds.  Instead, it extracts the exception pointer and performs a "normal"
// return.
//
// See also: rt/rust_try.ll

#[cfg(not(target_arch = "arm"), not(windows, target_arch = "x86_64"), not(test))]
#[doc(hidden)]
pub mod eabi {
    use libunwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    extern fn rust_eh_personality(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_v0(version, actions, exception_class, ue_header,
                                 context)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        _version: c_int,
        actions: uw::_Unwind_Action,
        _exception_class: uw::_Unwind_Exception_Class,
        _ue_header: *mut uw::_Unwind_Exception,
        _context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {

        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            uw::_URC_INSTALL_CONTEXT
        }
    }
}

// iOS on armv7 is using SjLj exceptions and therefore requires to use
// a specialized personality routine: __gcc_personality_sj0

#[cfg(target_os = "ios", target_arch = "arm", not(test))]
#[doc(hidden)]
pub mod eabi {
    use libunwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_sj0(version: c_int,
                                actions: uw::_Unwind_Action,
                                exception_class: uw::_Unwind_Exception_Class,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality(
        version: c_int,
        actions: uw::_Unwind_Action,
        exception_class: uw::_Unwind_Exception_Class,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_sj0(version, actions, exception_class, ue_header,
                                  context)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        _version: c_int,
        actions: uw::_Unwind_Action,
        _exception_class: uw::_Unwind_Exception_Class,
        _ue_header: *mut uw::_Unwind_Exception,
        _context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            unsafe {
                __gcc_personality_sj0(_version, actions, _exception_class, _ue_header,
                                      _context)
            }
        }
    }
}


// ARM EHABI uses a slightly different personality routine signature,
// but otherwise works the same.
#[cfg(target_arch = "arm", not(target_os = "ios"), not(test))]
#[doc(hidden)]
pub mod eabi {
    use libunwind as uw;
    use libc::c_int;

    extern "C" {
        fn __gcc_personality_v0(state: uw::_Unwind_State,
                                ue_header: *mut uw::_Unwind_Exception,
                                context: *mut uw::_Unwind_Context)
            -> uw::_Unwind_Reason_Code;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    extern "C" fn rust_eh_personality(
        state: uw::_Unwind_State,
        ue_header: *mut uw::_Unwind_Exception,
        context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        unsafe {
            __gcc_personality_v0(state, ue_header, context)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        state: uw::_Unwind_State,
        _ue_header: *mut uw::_Unwind_Exception,
        _context: *mut uw::_Unwind_Context
    ) -> uw::_Unwind_Reason_Code
    {
        if (state as c_int & uw::_US_ACTION_MASK as c_int)
                           == uw::_US_VIRTUAL_UNWIND_FRAME as c_int { // search phase
            uw::_URC_HANDLER_FOUND // catch!
        }
        else { // cleanup phase
            uw::_URC_INSTALL_CONTEXT
        }
    }
}

// Win64 SEH (see http://msdn.microsoft.com/en-us/library/1eyas8tf.aspx)
//
// This looks a bit convoluted because rather than implementing a native SEH handler,
// GCC reuses the same personality routine as for the other architectures by wrapping it
// with an "API translator" layer (_GCC_specific_handler).

#[cfg(windows, target_arch = "x86_64", not(test))]
#[doc(hidden)]
#[allow(non_camel_case_types, non_snake_case)]
pub mod eabi {
    use libunwind as uw;
    use libc::{c_void, c_int};

    #[repr(C)]
    pub struct EXCEPTION_RECORD;
    #[repr(C)]
    pub struct CONTEXT;
    #[repr(C)]
    pub struct DISPATCHER_CONTEXT;

    #[repr(C)]
    pub enum EXCEPTION_DISPOSITION {
        ExceptionContinueExecution,
        ExceptionContinueSearch,
        ExceptionNestedException,
        ExceptionCollidedUnwind
    }

    type _Unwind_Personality_Fn =
        extern "C" fn(
            version: c_int,
            actions: uw::_Unwind_Action,
            exception_class: uw::_Unwind_Exception_Class,
            ue_header: *mut uw::_Unwind_Exception,
            context: *mut uw::_Unwind_Context
        ) -> uw::_Unwind_Reason_Code;

    extern "C" {
        fn __gcc_personality_seh0(
            exceptionRecord: *mut EXCEPTION_RECORD,
            establisherFrame: *mut c_void,
            contextRecord: *mut CONTEXT,
            dispatcherContext: *mut DISPATCHER_CONTEXT
        ) -> EXCEPTION_DISPOSITION;

        fn _GCC_specific_handler(
            exceptionRecord: *mut EXCEPTION_RECORD,
            establisherFrame: *mut c_void,
            contextRecord: *mut CONTEXT,
            dispatcherContext: *mut DISPATCHER_CONTEXT,
            personality: _Unwind_Personality_Fn
        ) -> EXCEPTION_DISPOSITION;
    }

    #[lang="eh_personality"]
    #[no_mangle] // referenced from rust_try.ll
    extern "C" fn rust_eh_personality(
        exceptionRecord: *mut EXCEPTION_RECORD,
        establisherFrame: *mut c_void,
        contextRecord: *mut CONTEXT,
        dispatcherContext: *mut DISPATCHER_CONTEXT
    ) -> EXCEPTION_DISPOSITION
    {
        unsafe {
            __gcc_personality_seh0(exceptionRecord, establisherFrame,
                                   contextRecord, dispatcherContext)
        }
    }

    #[no_mangle] // referenced from rust_try.ll
    pub extern "C" fn rust_eh_personality_catch(
        exceptionRecord: *mut EXCEPTION_RECORD,
        establisherFrame: *mut c_void,
        contextRecord: *mut CONTEXT,
        dispatcherContext: *mut DISPATCHER_CONTEXT
    ) -> EXCEPTION_DISPOSITION
    {
        extern "C" fn inner(
                _version: c_int,
                actions: uw::_Unwind_Action,
                _exception_class: uw::_Unwind_Exception_Class,
                _ue_header: *mut uw::_Unwind_Exception,
                _context: *mut uw::_Unwind_Context
            ) -> uw::_Unwind_Reason_Code
        {
            if (actions as c_int & uw::_UA_SEARCH_PHASE as c_int) != 0 { // search phase
                uw::_URC_HANDLER_FOUND // catch!
            }
            else { // cleanup phase
                uw::_URC_INSTALL_CONTEXT
            }
        }

        unsafe {
            _GCC_specific_handler(exceptionRecord, establisherFrame,
                                  contextRecord, dispatcherContext,
                                  inner)
        }
    }
}

// Entry point of failure from the libcore crate
#[cfg(not(test))]
#[lang = "fail_fmt"]
pub extern fn rust_begin_unwind(msg: &fmt::Arguments,
                                file: &'static str, line: uint) -> ! {
    begin_unwind_fmt(msg, &(file, line))
}

/// The entry point for unwinding with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `fail!()` has as low an impact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[inline(never)] #[cold]
pub fn begin_unwind_fmt(msg: &fmt::Arguments, file_line: &(&'static str, uint)) -> ! {
    use core::fmt::FormatWriter;

    // We do two allocations here, unfortunately. But (a) they're
    // required with the current scheme, and (b) we don't handle
    // failure + OOM properly anyway (see comment in begin_unwind
    // below).

    struct VecWriter<'a> { v: &'a mut Vec<u8> }

    impl<'a> fmt::FormatWriter for VecWriter<'a> {
        fn write(&mut self, buf: &[u8]) -> fmt::Result {
            self.v.push_all(buf);
            Ok(())
        }
    }

    let mut v = Vec::new();
    let _ = write!(&mut VecWriter { v: &mut v }, "{}", msg);

    let msg = box String::from_utf8_lossy(v.as_slice()).into_string();
    begin_unwind_inner(msg, file_line)
}

/// This is the entry point of unwinding for fail!() and assert!().
#[inline(never)] #[cold] // avoid code bloat at the call sites as much as possible
pub fn begin_unwind<M: Any + Send>(msg: M, file_line: &(&'static str, uint)) -> ! {
    // Note that this should be the only allocation performed in this code path.
    // Currently this means that fail!() on OOM will invoke this code path,
    // but then again we're not really ready for failing on OOM anyway. If
    // we do start doing this, then we should propagate this allocation to
    // be performed in the parent of this task instead of the task that's
    // failing.

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
/// Do this split took the LLVM IR line counts of `fn main() { fail!()
/// }` from ~1900/3700 (-O/no opts) to 180/590.
#[inline(never)] #[cold] // this is the slow path, please never inline this
fn begin_unwind_inner(msg: Box<Any + Send>, file_line: &(&'static str, uint)) -> ! {
    // First, invoke call the user-defined callbacks triggered on task failure.
    //
    // By the time that we see a callback has been registered (by reading
    // MAX_CALLBACKS), the actual callback itself may have not been stored yet,
    // so we just chalk it up to a race condition and move on to the next
    // callback. Additionally, CALLBACK_CNT may briefly be higher than
    // MAX_CALLBACKS, so we're sure to clamp it as necessary.
    let callbacks = unsafe {
        let amt = CALLBACK_CNT.load(atomic::SeqCst);
        CALLBACKS.slice_to(cmp::min(amt, MAX_CALLBACKS))
    };
    for cb in callbacks.iter() {
        match cb.load(atomic::SeqCst) {
            0 => {}
            n => {
                let f: Callback = unsafe { mem::transmute(n) };
                let (file, line) = *file_line;
                f(&*msg, file, line);
            }
        }
    };

    // Now that we've run all the necessary unwind callbacks, we actually
    // perform the unwinding. If we don't have a task, then it's time to die
    // (hopefully someone printed something about this).
    let mut task: Box<Task> = match Local::try_take() {
        Some(task) => task,
        None => rust_fail(msg),
    };

    if task.unwinder.unwinding {
        // If a task fails while it's already unwinding then we
        // have limited options. Currently our preference is to
        // just abort. In the future we may consider resuming
        // unwinding or otherwise exiting the task cleanly.
        rterrln!("task failed during unwinding. aborting.");
        unsafe { intrinsics::abort() }
    }
    task.unwinder.unwinding = true;

    // Put the task back in TLS because the unwinding process may run code which
    // requires the task. We need a handle to its unwinder, however, so after
    // this we unsafely extract it and continue along.
    Local::put(task);
    rust_fail(msg);
}

/// Register a callback to be invoked when a task unwinds.
///
/// This is an unsafe and experimental API which allows for an arbitrary
/// callback to be invoked when a task fails. This callback is invoked on both
/// the initial unwinding and a double unwinding if one occurs. Additionally,
/// the local `Task` will be in place for the duration of the callback, and
/// the callback must ensure that it remains in place once the callback returns.
///
/// Only a limited number of callbacks can be registered, and this function
/// returns whether the callback was successfully registered or not. It is not
/// currently possible to unregister a callback once it has been registered.
#[experimental]
pub unsafe fn register(f: Callback) -> bool {
    match CALLBACK_CNT.fetch_add(1, atomic::SeqCst) {
        // The invocation code has knowledge of this window where the count has
        // been incremented, but the callback has not been stored. We're
        // guaranteed that the slot we're storing into is 0.
        n if n < MAX_CALLBACKS => {
            let prev = CALLBACKS[n].swap(mem::transmute(f), atomic::SeqCst);
            rtassert!(prev == 0);
            true
        }
        // If we accidentally bumped the count too high, pull it back.
        _ => {
            CALLBACK_CNT.store(MAX_CALLBACKS, atomic::SeqCst);
            false
        }
    }
}
