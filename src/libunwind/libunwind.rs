// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(bad_style)]

use libc;

#[cfg(any(not(target_arch = "arm"), target_os = "ios"))]
pub use self::_Unwind_Action::*;
#[cfg(target_arch = "arm")]
pub use self::_Unwind_State::*;
pub use self::_Unwind_Reason_Code::*;

#[cfg(any(not(target_arch = "arm"), target_os = "ios"))]
#[repr(C)]
#[derive(Clone, Copy)]
pub enum _Unwind_Action {
    _UA_SEARCH_PHASE = 1,
    _UA_CLEANUP_PHASE = 2,
    _UA_HANDLER_FRAME = 4,
    _UA_FORCE_UNWIND = 8,
    _UA_END_OF_STACK = 16,
}

#[cfg(target_arch = "arm")]
#[repr(C)]
#[derive(Clone, Copy)]
pub enum _Unwind_State {
    _US_VIRTUAL_UNWIND_FRAME = 0,
    _US_UNWIND_FRAME_STARTING = 1,
    _US_UNWIND_FRAME_RESUME = 2,
    _US_ACTION_MASK = 3,
    _US_FORCE_UNWIND = 8,
    _US_END_OF_STACK = 16,
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

pub type _Unwind_Word = libc::uintptr_t;

pub type _Unwind_Trace_Fn = extern "C" fn(ctx: *mut _Unwind_Context, arg: *mut libc::c_void)
                                          -> _Unwind_Reason_Code;

#[cfg(target_arch = "x86")]
pub const unwinder_private_data_size: usize = 5;

#[cfg(target_arch = "x86_64")]
pub const unwinder_private_data_size: usize = 6;

#[cfg(all(target_arch = "arm", not(target_os = "ios")))]
pub const unwinder_private_data_size: usize = 20;

#[cfg(all(target_arch = "arm", target_os = "ios"))]
pub const unwinder_private_data_size: usize = 5;

#[cfg(target_arch = "aarch64")]
pub const unwinder_private_data_size: usize = 2;

#[cfg(target_arch = "mips")]
pub const unwinder_private_data_size: usize = 2;

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(target_arch = "asmjs")]
// FIXME: Copied from arm. Need to confirm.
pub const unwinder_private_data_size: usize = 20;

#[repr(C)]
pub struct _Unwind_Exception {
    pub exception_class: _Unwind_Exception_Class,
    pub exception_cleanup: _Unwind_Exception_Cleanup_Fn,
    pub private: [_Unwind_Word; unwinder_private_data_size],
}

pub enum _Unwind_Context {}

pub type _Unwind_Exception_Cleanup_Fn = extern "C" fn(unwind_code: _Unwind_Reason_Code,
                                                      exception: *mut _Unwind_Exception);

#[cfg_attr(any(all(target_os = "linux", not(target_env = "musl")),
               target_os = "freebsd",
               target_os = "solaris",
               all(target_os = "linux",
                   target_env = "musl",
                   not(target_arch = "x86"),
                   not(target_arch = "x86_64"))),
           link(name = "gcc_s"))]
#[cfg_attr(all(target_os = "linux",
               target_env = "musl",
               any(target_arch = "x86", target_arch = "x86_64"),
               not(test)),
           link(name = "unwind", kind = "static"))]
#[cfg_attr(any(target_os = "android", target_os = "openbsd"),
           link(name = "gcc"))]
#[cfg_attr(all(target_os = "netbsd", not(target_vendor = "rumprun")),
           link(name = "gcc"))]
#[cfg_attr(all(target_os = "netbsd", target_vendor = "rumprun"),
           link(name = "unwind"))]
#[cfg_attr(target_os = "dragonfly",
           link(name = "gcc_pic"))]
#[cfg_attr(target_os = "bitrig",
           link(name = "c++abi"))]
#[cfg_attr(all(target_os = "windows", target_env = "gnu"),
           link(name = "gcc_eh"))]
#[cfg(not(cargobuild))]
extern "C" {}

extern "C" {
    // iOS on armv7 uses SjLj exceptions and requires to link
    // against corresponding routine (..._SjLj_...)
    #[cfg(not(all(target_os = "ios", target_arch = "arm")))]
    #[unwind]
    pub fn _Unwind_RaiseException(exception: *mut _Unwind_Exception) -> _Unwind_Reason_Code;

    #[cfg(all(target_os = "ios", target_arch = "arm"))]
    #[unwind]
    fn _Unwind_SjLj_RaiseException(e: *mut _Unwind_Exception) -> _Unwind_Reason_Code;

    pub fn _Unwind_DeleteException(exception: *mut _Unwind_Exception);

    #[unwind]
    pub fn _Unwind_Resume(exception: *mut _Unwind_Exception) -> !;

    // No native _Unwind_Backtrace on iOS
    #[cfg(not(all(target_os = "ios", target_arch = "arm")))]
    pub fn _Unwind_Backtrace(trace: _Unwind_Trace_Fn,
                             trace_argument: *mut libc::c_void)
                             -> _Unwind_Reason_Code;

    // available since GCC 4.2.0, should be fine for our purpose
    #[cfg(all(not(all(target_os = "android", target_arch = "arm")),
              not(all(target_os = "linux", target_arch = "arm"))))]
    pub fn _Unwind_GetIPInfo(ctx: *mut _Unwind_Context,
                             ip_before_insn: *mut libc::c_int)
                             -> libc::uintptr_t;

    #[cfg(all(not(target_os = "android"),
              not(all(target_os = "linux", target_arch = "arm"))))]
    pub fn _Unwind_FindEnclosingFunction(pc: *mut libc::c_void) -> *mut libc::c_void;
}

// ... and now we just providing access to SjLj counterspart
// through a standard name to hide those details from others
// (see also comment above regarding _Unwind_RaiseException)
#[cfg(all(target_os = "ios", target_arch = "arm"))]
#[inline]
pub unsafe fn _Unwind_RaiseException(exc: *mut _Unwind_Exception) -> _Unwind_Reason_Code {
    _Unwind_SjLj_RaiseException(exc)
}

// On android, the function _Unwind_GetIP is a macro, and this is the
// expansion of the macro. This is all copy/pasted directly from the
// header file with the definition of _Unwind_GetIP.
#[cfg(any(all(target_os = "android", target_arch = "arm"),
          all(target_os = "linux", target_arch = "arm")))]
pub unsafe fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> libc::uintptr_t {
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

    type _Unwind_Word = libc::c_uint;
    extern "C" {
        fn _Unwind_VRS_Get(ctx: *mut _Unwind_Context,
                           klass: _Unwind_VRS_RegClass,
                           word: _Unwind_Word,
                           repr: _Unwind_VRS_DataRepresentation,
                           data: *mut libc::c_void)
                           -> _Unwind_VRS_Result;
    }

    let mut val: _Unwind_Word = 0;
    let ptr = &mut val as *mut _Unwind_Word;
    let _ = _Unwind_VRS_Get(ctx,
                            _Unwind_VRS_RegClass::_UVRSC_CORE,
                            15,
                            _Unwind_VRS_DataRepresentation::_UVRSD_UINT32,
                            ptr as *mut libc::c_void);
    (val & !1) as libc::uintptr_t
}

// This function doesn't exist on Android or ARM/Linux, so make it same
// to _Unwind_GetIP
#[cfg(any(all(target_os = "android", target_arch = "arm"),
          all(target_os = "linux", target_arch = "arm")))]
pub unsafe fn _Unwind_GetIPInfo(ctx: *mut _Unwind_Context,
                                ip_before_insn: *mut libc::c_int)
                                -> libc::uintptr_t {
    *ip_before_insn = 0;
    _Unwind_GetIP(ctx)
}

// This function also doesn't exist on Android or ARM/Linux, so make it
// a no-op
#[cfg(any(target_os = "android",
          all(target_os = "linux", target_arch = "arm")))]
pub unsafe fn _Unwind_FindEnclosingFunction(pc: *mut libc::c_void) -> *mut libc::c_void {
    pc
}
