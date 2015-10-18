// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unwind library interface

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)] // these are just bindings

#[cfg(any(not(target_arch = "arm"), target_os = "ios"))]
pub use self::_Unwind_Action::*;
#[cfg(target_arch = "arm")]
pub use self::_Unwind_State::*;
pub use self::_Unwind_Reason_Code::*;

use libc;

#[cfg(any(not(target_arch = "arm"), target_os = "ios"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub enum _Unwind_Action {
    _UA_SEARCH_PHASE = 1,
    _UA_CLEANUP_PHASE = 2,
    _UA_HANDLER_FRAME = 4,
    _UA_FORCE_UNWIND = 8,
    _UA_END_OF_STACK = 16,
}

#[cfg(target_arch = "arm")]
#[repr(C)]
#[derive(Copy, Clone)]
pub enum _Unwind_State {
    _US_VIRTUAL_UNWIND_FRAME = 0,
    _US_UNWIND_FRAME_STARTING = 1,
    _US_UNWIND_FRAME_RESUME = 2,
    _US_ACTION_MASK = 3,
    _US_FORCE_UNWIND = 8,
    _US_END_OF_STACK = 16
}

#[repr(C)]
#[derive(Copy, Clone)]
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
pub const unwinder_private_data_size: usize = 5;

#[cfg(target_arch = "x86_64")]
pub const unwinder_private_data_size: usize = 6;

#[cfg(all(target_arch = "arm", not(target_os = "ios")))]
pub const unwinder_private_data_size: usize = 20;

#[cfg(all(target_arch = "arm", target_os = "ios"))]
pub const unwinder_private_data_size: usize = 5;

#[cfg(target_arch = "aarch64")]
pub const unwinder_private_data_size: usize = 2;

#[cfg(any(target_arch = "mips", target_arch = "mipsel"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(target_arch = "powerpc")]
pub const unwinder_private_data_size: usize = 2;

#[repr(C)]
pub struct _Unwind_Exception {
    pub exception_class: _Unwind_Exception_Class,
    pub exception_cleanup: _Unwind_Exception_Cleanup_Fn,
    pub private: [_Unwind_Word; unwinder_private_data_size],
}

pub enum _Unwind_Context {}

pub type _Unwind_Exception_Cleanup_Fn =
        extern "C" fn(unwind_code: _Unwind_Reason_Code,
                      exception: *mut _Unwind_Exception);

#[cfg_attr(any(all(target_os = "linux", not(target_env = "musl")),
               target_os = "freebsd"),
           link(name = "gcc_s"))]
#[cfg_attr(all(target_os = "linux", target_env = "musl", not(test)),
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
#[cfg_attr(all(target_os = "windows", target_env="gnu"),
           link(name = "gcc_eh"))]
extern "C" {
    // iOS on armv7 uses SjLj exceptions and requires to link
    // against corresponding routine (..._SjLj_...)
    #[cfg(not(all(target_os = "ios", target_arch = "arm")))]
    #[unwind]
    pub fn _Unwind_RaiseException(exception: *mut _Unwind_Exception)
                                  -> _Unwind_Reason_Code;

    #[cfg(all(target_os = "ios", target_arch = "arm"))]
    #[unwind]
    fn _Unwind_SjLj_RaiseException(e: *mut _Unwind_Exception)
                                   -> _Unwind_Reason_Code;

    pub fn _Unwind_DeleteException(exception: *mut _Unwind_Exception);

    #[unwind]
    pub fn _Unwind_Resume(exception: *mut _Unwind_Exception) -> !;
}

// ... and now we just providing access to SjLj counterspart
// through a standard name to hide those details from others
// (see also comment above regarding _Unwind_RaiseException)
#[cfg(all(target_os = "ios", target_arch = "arm"))]
#[inline(always)]
pub unsafe fn _Unwind_RaiseException(exc: *mut _Unwind_Exception)
                                     -> _Unwind_Reason_Code {
    _Unwind_SjLj_RaiseException(exc)
}
