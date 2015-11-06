// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use io::prelude::*;
use libc;
use libc::{c_void};
use mem;
use sync::StaticMutex;

use super::super::printing::print;

#[inline(never)] // if we know this is a function call, we can skip it when
                 // tracing
pub fn write(w: &mut Write) -> io::Result<()> {
    struct Context<'a> {
        idx: isize,
        writer: &'a mut (Write+'a),
        last_error: Option<io::Error>,
    }

    // When using libbacktrace, we use some necessary global state, so we
    // need to prevent more than one thread from entering this block. This
    // is semi-reasonable in terms of printing anyway, and we know that all
    // I/O done here is blocking I/O, not green I/O, so we don't have to
    // worry about this being a native vs green mutex.
    static LOCK: StaticMutex = StaticMutex::new();
    let _g = LOCK.lock();

    try!(writeln!(w, "stack backtrace:"));

    let mut cx = Context { writer: w, last_error: None, idx: 0 };
    return match unsafe {
        uw::_Unwind_Backtrace(trace_fn,
                              &mut cx as *mut Context as *mut c_void)
    } {
        uw::_URC_NO_REASON => {
            match cx.last_error {
                Some(err) => Err(err),
                None => Ok(())
            }
        }
        _ => Ok(()),
    };

    extern fn trace_fn(ctx: *mut uw::_Unwind_Context,
                       arg: *mut c_void) -> uw::_Unwind_Reason_Code {
        let cx: &mut Context = unsafe { mem::transmute(arg) };
        let mut ip_before_insn = 0;
        let mut ip = unsafe {
            uw::_Unwind_GetIPInfo(ctx, &mut ip_before_insn) as *mut c_void
        };
        if !ip.is_null() && ip_before_insn == 0 {
            // this is a non-signaling frame, so `ip` refers to the address
            // after the calling instruction. account for that.
            ip = (ip as usize - 1) as *mut _;
        }

        // dladdr() on osx gets whiny when we use FindEnclosingFunction, and
        // it appears to work fine without it, so we only use
        // FindEnclosingFunction on non-osx platforms. In doing so, we get a
        // slightly more accurate stack trace in the process.
        //
        // This is often because panic involves the last instruction of a
        // function being "call std::rt::begin_unwind", with no ret
        // instructions after it. This means that the return instruction
        // pointer points *outside* of the calling function, and by
        // unwinding it we go back to the original function.
        let symaddr = if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
            ip
        } else {
            unsafe { uw::_Unwind_FindEnclosingFunction(ip) }
        };

        // Don't print out the first few frames (they're not user frames)
        cx.idx += 1;
        if cx.idx <= 0 { return uw::_URC_NO_REASON }
        // Don't print ginormous backtraces
        if cx.idx > 100 {
            match write!(cx.writer, " ... <frames omitted>\n") {
                Ok(()) => {}
                Err(e) => { cx.last_error = Some(e); }
            }
            return uw::_URC_FAILURE
        }

        // Once we hit an error, stop trying to print more frames
        if cx.last_error.is_some() { return uw::_URC_FAILURE }

        match print(cx.writer, cx.idx, ip, symaddr) {
            Ok(()) => {}
            Err(e) => { cx.last_error = Some(e); }
        }

        // keep going
        uw::_URC_NO_REASON
    }
}

/// Unwind library interface used for backtraces
///
/// Note that dead code is allowed as here are just bindings
/// iOS doesn't use all of them it but adding more
/// platform-specific configs pollutes the code too much
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
mod uw {
    pub use self::_Unwind_Reason_Code::*;

    use libc;
    use libc::{c_int, c_uint, c_void, uintptr_t};

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

    pub enum _Unwind_Context {}

    pub type _Unwind_Trace_Fn =
            extern fn(ctx: *mut _Unwind_Context,
                      arg: *mut c_void) -> _Unwind_Reason_Code;

    extern {
        // No native _Unwind_Backtrace on iOS
        #[cfg(not(all(target_os = "ios", target_arch = "arm")))]
        pub fn _Unwind_Backtrace(trace: _Unwind_Trace_Fn,
                                 trace_argument: *mut c_void)
                    -> _Unwind_Reason_Code;

        // available since GCC 4.2.0, should be fine for our purpose
        #[cfg(all(not(all(target_os = "android", target_arch = "arm")),
                  not(all(target_os = "linux", target_arch = "arm"))))]
        pub fn _Unwind_GetIPInfo(ctx: *mut _Unwind_Context,
                                 ip_before_insn: *mut c_int)
                    -> uintptr_t;

        #[cfg(all(not(target_os = "android"),
                  not(all(target_os = "linux", target_arch = "arm"))))]
        pub fn _Unwind_FindEnclosingFunction(pc: *mut c_void)
            -> *mut c_void;
    }

    // On android, the function _Unwind_GetIP is a macro, and this is the
    // expansion of the macro. This is all copy/pasted directly from the
    // header file with the definition of _Unwind_GetIP.
    #[cfg(any(all(target_os = "android", target_arch = "arm"),
              all(target_os = "linux", target_arch = "arm")))]
    pub unsafe fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> uintptr_t {
        #[repr(C)]
        enum _Unwind_VRS_Result {
            _UVRSR_OK = 0,
            _UVRSR_NOT_IMPLEMENTED = 1,
            _UVRSR_FAILED = 2,
        }
        #[repr(C)]
        enum _Unwind_VRS_RegClass {
            _UVRSC_CORE = 0,
            _UVRSC_VFP = 1,
            _UVRSC_FPA = 2,
            _UVRSC_WMMXD = 3,
            _UVRSC_WMMXC = 4,
        }
        #[repr(C)]
        enum _Unwind_VRS_DataRepresentation {
            _UVRSD_UINT32 = 0,
            _UVRSD_VFPX = 1,
            _UVRSD_FPAX = 2,
            _UVRSD_UINT64 = 3,
            _UVRSD_FLOAT = 4,
            _UVRSD_DOUBLE = 5,
        }

        type _Unwind_Word = c_uint;
        extern {
            fn _Unwind_VRS_Get(ctx: *mut _Unwind_Context,
                               klass: _Unwind_VRS_RegClass,
                               word: _Unwind_Word,
                               repr: _Unwind_VRS_DataRepresentation,
                               data: *mut c_void)
                -> _Unwind_VRS_Result;
        }

        let mut val: _Unwind_Word = 0;
        let ptr = &mut val as *mut _Unwind_Word;
        let _ = _Unwind_VRS_Get(ctx, _Unwind_VRS_RegClass::_UVRSC_CORE, 15,
                                _Unwind_VRS_DataRepresentation::_UVRSD_UINT32,
                                ptr as *mut c_void);
        (val & !1) as uintptr_t
    }

    // This function doesn't exist on Android or ARM/Linux, so make it same
    // to _Unwind_GetIP
    #[cfg(any(all(target_os = "android", target_arch = "arm"),
              all(target_os = "linux", target_arch = "arm")))]
    pub unsafe fn _Unwind_GetIPInfo(ctx: *mut _Unwind_Context,
                                    ip_before_insn: *mut c_int)
        -> uintptr_t
    {
        *ip_before_insn = 0;
        _Unwind_GetIP(ctx)
    }

    // This function also doesn't exist on Android or ARM/Linux, so make it
    // a no-op
    #[cfg(any(target_os = "android",
              all(target_os = "linux", target_arch = "arm")))]
    pub unsafe fn _Unwind_FindEnclosingFunction(pc: *mut c_void)
        -> *mut c_void
    {
        pc
    }
}
