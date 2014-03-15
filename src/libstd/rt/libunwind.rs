// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unwind library interface

#[allow(non_camel_case_types)];
#[allow(dead_code)]; // these are just bindings

use libc;

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

pub type _Unwind_Word = libc::uintptr_t;

#[cfg(target_arch = "x86")]
pub static unwinder_private_data_size: int = 5;

#[cfg(target_arch = "x86_64")]
pub static unwinder_private_data_size: int = 2;

#[cfg(target_arch = "arm")]
pub static unwinder_private_data_size: int = 20;

#[cfg(target_arch = "mips")]
pub static unwinder_private_data_size: int = 2;

pub struct _Unwind_Exception {
    exception_class: _Unwind_Exception_Class,
    exception_cleanup: _Unwind_Exception_Cleanup_Fn,
    private: [_Unwind_Word, ..unwinder_private_data_size],
}

pub enum _Unwind_Context {}

pub type _Unwind_Exception_Cleanup_Fn =
        extern "C" fn(unwind_code: _Unwind_Reason_Code,
                      exception: *_Unwind_Exception);

pub type _Unwind_Trace_Fn =
        extern "C" fn(ctx: *_Unwind_Context,
                      arg: *libc::c_void) -> _Unwind_Reason_Code;

#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "win32")]
#[link(name = "gcc_s")]
extern {}

#[cfg(target_os = "android")]
#[link(name = "gcc")]
extern {}

extern "C" {
    pub fn _Unwind_RaiseException(exception: *_Unwind_Exception)
                -> _Unwind_Reason_Code;
    pub fn _Unwind_DeleteException(exception: *_Unwind_Exception);
    pub fn _Unwind_Backtrace(trace: _Unwind_Trace_Fn,
                             trace_argument: *libc::c_void)
                -> _Unwind_Reason_Code;
    #[cfg(stage0, not(target_os = "android"))]
    pub fn _Unwind_GetIP(ctx: *_Unwind_Context) -> libc::uintptr_t;
    #[cfg(stage0, not(target_os = "android"))]
    pub fn _Unwind_FindEnclosingFunction(pc: *libc::c_void) -> *libc::c_void;

    #[cfg(not(stage0),
          not(target_os = "android"),
          not(target_os = "linux", target_arch = "arm"))]
    pub fn _Unwind_GetIP(ctx: *_Unwind_Context) -> libc::uintptr_t;
    #[cfg(not(stage0),
          not(target_os = "android"),
          not(target_os = "linux", target_arch = "arm"))]
    pub fn _Unwind_FindEnclosingFunction(pc: *libc::c_void) -> *libc::c_void;
}

// On android, the function _Unwind_GetIP is a macro, and this is the expansion
// of the macro. This is all copy/pasted directly from the header file with the
// definition of _Unwind_GetIP.
#[cfg(target_os = "android")]
#[cfg(target_os = "linux", target_os = "arm")]
pub unsafe fn _Unwind_GetIP(ctx: *_Unwind_Context) -> libc::uintptr_t {
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
    extern {
        fn _Unwind_VRS_Get(ctx: *_Unwind_Context,
                           klass: _Unwind_VRS_RegClass,
                           word: _Unwind_Word,
                           repr: _Unwind_VRS_DataRepresentation,
                           data: *mut libc::c_void) -> _Unwind_VRS_Result;
    }

    let mut val: _Unwind_Word = 0;
    let ptr = &mut val as *mut _Unwind_Word;
    let _ = _Unwind_VRS_Get(ctx, _UVRSC_CORE, 15, _UVRSD_UINT32,
                            ptr as *mut libc::c_void);
    (val & !1) as libc::uintptr_t
}

// This function also doesn't exist on android, so make it a no-op
#[cfg(target_os = "android")]
#[cfg(target_os = "linux", target_os = "arm")]
pub unsafe fn _Unwind_FindEnclosingFunction(pc: *libc::c_void) -> *libc::c_void{
    pc
}
