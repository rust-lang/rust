// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unwinding implementation of top of native Win64 SEH,
//! however the unwind handler data (aka LSDA) uses GCC-compatible encoding.

#![allow(bad_style)]
#![allow(private_no_mangle_fns)]

use alloc::boxed::Box;

use core::any::Any;
use core::intrinsics;
use core::ptr;
use dwarf::eh::{EHContext, EHAction, find_eh_action};
use windows as c;

// Define our exception codes:
// according to http://msdn.microsoft.com/en-us/library/het71c37(v=VS.80).aspx,
//    [31:30] = 3 (error), 2 (warning), 1 (info), 0 (success)
//    [29]    = 1 (user-defined)
//    [28]    = 0 (reserved)
// we define bits:
//    [24:27] = type
//    [0:23]  = magic
const ETYPE: c::DWORD = 0b1110_u32 << 28;
const MAGIC: c::DWORD = 0x525354; // "RST"

const RUST_PANIC: c::DWORD = ETYPE | (1 << 24) | MAGIC;

#[repr(C)]
struct PanicData {
    data: Box<Any + Send>,
}

pub unsafe fn panic(data: Box<Any + Send>) -> u32 {
    let panic_ctx = Box::new(PanicData { data: data });
    let params = [Box::into_raw(panic_ctx) as c::ULONG_PTR];
    c::RaiseException(RUST_PANIC,
                      c::EXCEPTION_NONCONTINUABLE,
                      params.len() as c::DWORD,
                      &params as *const c::ULONG_PTR);
    u32::max_value()
}

pub fn payload() -> *mut u8 {
    ptr::null_mut()
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<Any + Send> {
    let panic_ctx = Box::from_raw(ptr as *mut PanicData);
    return panic_ctx.data;
}

// SEH doesn't support resuming unwinds after calling a landing pad like
// libunwind does. For this reason, MSVC compiler outlines landing pads into
// separate functions that can be called directly from the personality function
// but are nevertheless able to find and modify stack frame of the "parent"
// function.
//
// Since this cannot be done with libdwarf-style landing pads,
// rust_eh_personality instead catches RUST_PANICs, runs the landing pad, then
// reraises the exception.
//
// Note that it makes certain assumptions about the exception:
//
// 1. That RUST_PANIC is non-continuable, so no lower stack frame may choose to
//    resume execution.
// 2. That the first parameter of the exception is a pointer to an extra data
//    area (PanicData).
// Since these assumptions do not generally hold true for foreign exceptions
// (system faults, C++ exceptions, etc), we make no attempt to invoke our
// landing pads (and, thus, destructors!) for anything other than RUST_PANICs.
// This is considered acceptable, because the behavior of throwing exceptions
// through a C ABI boundary is undefined.

#[lang = "eh_personality"]
#[cfg(not(test))]
unsafe extern "C" fn rust_eh_personality(exceptionRecord: *mut c::EXCEPTION_RECORD,
                                         establisherFrame: c::LPVOID,
                                         contextRecord: *mut c::CONTEXT,
                                         dispatcherContext: *mut c::DISPATCHER_CONTEXT)
                                         -> c::EXCEPTION_DISPOSITION {
    let er = &*exceptionRecord;
    let dc = &*dispatcherContext;

    if er.ExceptionFlags & c::EXCEPTION_UNWIND == 0 {
        // we are in the dispatch phase
        if er.ExceptionCode == RUST_PANIC {
            if let Some(lpad) = find_landing_pad(dc) {
                c::RtlUnwindEx(establisherFrame,
                               lpad as c::LPVOID,
                               exceptionRecord,
                               er.ExceptionInformation[0] as c::LPVOID, // pointer to PanicData
                               contextRecord,
                               dc.HistoryTable);
            }
        }
    }
    c::ExceptionContinueSearch
}

#[lang = "eh_unwind_resume"]
#[unwind]
unsafe extern "C" fn rust_eh_unwind_resume(panic_ctx: c::LPVOID) -> ! {
    let params = [panic_ctx as c::ULONG_PTR];
    c::RaiseException(RUST_PANIC,
                      c::EXCEPTION_NONCONTINUABLE,
                      params.len() as c::DWORD,
                      &params as *const c::ULONG_PTR);
    intrinsics::abort();
}

unsafe fn find_landing_pad(dc: &c::DISPATCHER_CONTEXT) -> Option<usize> {
    let eh_ctx = EHContext {
        // The return address points 1 byte past the call instruction,
        // which could be in the next IP range in LSDA range table.
        ip: dc.ControlPc as usize - 1,
        func_start: dc.ImageBase as usize + (*dc.FunctionEntry).BeginAddress as usize,
        get_text_start: &|| dc.ImageBase as usize,
        get_data_start: &|| unimplemented!(),
    };
    match find_eh_action(dc.HandlerData, &eh_ctx) {
        EHAction::None => None,
        EHAction::Cleanup(lpad) |
        EHAction::Catch(lpad) => Some(lpad),
        EHAction::Terminate => intrinsics::abort(),
    }
}
