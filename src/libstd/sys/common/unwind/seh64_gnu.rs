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

use prelude::v1::*;

use any::Any;
use self::EXCEPTION_DISPOSITION::*;
use sys::common::dwarf::eh;
use core::mem;
use core::ptr;
use libc::{c_void, c_ulonglong, DWORD, LPVOID};
type ULONG_PTR = c_ulonglong;

// Define our exception codes:
// according to http://msdn.microsoft.com/en-us/library/het71c37(v=VS.80).aspx,
//    [31:30] = 3 (error), 2 (warning), 1 (info), 0 (success)
//    [29]    = 1 (user-defined)
//    [28]    = 0 (reserved)
// we define bits:
//    [24:27] = type
//    [0:23]  = magic
const ETYPE: DWORD = 0b1110_u32 << 28;
const MAGIC: DWORD = 0x525354; // "RST"

const RUST_PANIC: DWORD  = ETYPE | (1 << 24) | MAGIC;

const EXCEPTION_NONCONTINUABLE: DWORD = 0x1;   // Noncontinuable exception
const EXCEPTION_UNWINDING: DWORD = 0x2;        // Unwind is in progress
const EXCEPTION_EXIT_UNWIND: DWORD = 0x4;      // Exit unwind is in progress
const EXCEPTION_STACK_INVALID: DWORD = 0x8;    // Stack out of limits or unaligned
const EXCEPTION_NESTED_CALL: DWORD = 0x10;     // Nested exception handler call
const EXCEPTION_TARGET_UNWIND: DWORD = 0x20;   // Target unwind in progress
const EXCEPTION_COLLIDED_UNWIND: DWORD = 0x40; // Collided exception handler call
const EXCEPTION_UNWIND: DWORD = EXCEPTION_UNWINDING |
                                EXCEPTION_EXIT_UNWIND |
                                EXCEPTION_TARGET_UNWIND |
                                EXCEPTION_COLLIDED_UNWIND;

#[repr(C)]
pub struct EXCEPTION_RECORD {
    ExceptionCode: DWORD,
    ExceptionFlags: DWORD,
    ExceptionRecord: *const EXCEPTION_RECORD,
    ExceptionAddress: LPVOID,
    NumberParameters: DWORD,
    ExceptionInformation: [ULONG_PTR; 15],
}

pub enum CONTEXT {}
pub enum UNWIND_HISTORY_TABLE {}

#[repr(C)]
pub struct RUNTIME_FUNCTION {
    BeginAddress: DWORD,
    EndAddress: DWORD,
    UnwindData: DWORD,
}

#[repr(C)]
pub struct DISPATCHER_CONTEXT {
    ControlPc: LPVOID,
    ImageBase: LPVOID,
    FunctionEntry: *const RUNTIME_FUNCTION,
    EstablisherFrame: LPVOID,
    TargetIp: LPVOID,
    ContextRecord: *const CONTEXT,
    LanguageHandler: LPVOID,
    HandlerData: *const u8,
    HistoryTable: *const UNWIND_HISTORY_TABLE,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum EXCEPTION_DISPOSITION {
    ExceptionContinueExecution,
    ExceptionContinueSearch,
    ExceptionNestedException,
    ExceptionCollidedUnwind
}

// From kernel32.dll
extern "system" {
    #[unwind]
    fn RaiseException(dwExceptionCode: DWORD,
                      dwExceptionFlags: DWORD,
                      nNumberOfArguments: DWORD,
                      lpArguments: *const ULONG_PTR);

    fn RtlUnwindEx(TargetFrame: LPVOID,
                   TargetIp: LPVOID,
                   ExceptionRecord: *const EXCEPTION_RECORD,
                   ReturnValue: LPVOID,
                   OriginalContext: *const CONTEXT,
                   HistoryTable: *const UNWIND_HISTORY_TABLE);
}

#[repr(C)]
struct PanicData {
    data: Box<Any + Send + 'static>
}

pub unsafe fn panic(data: Box<Any + Send + 'static>) -> ! {
    let panic_ctx = Box::new(PanicData { data: data });
    let params = [Box::into_raw(panic_ctx) as ULONG_PTR];
    RaiseException(RUST_PANIC,
                   EXCEPTION_NONCONTINUABLE,
                   params.len() as DWORD,
                   &params as *const ULONG_PTR);
    rtabort!("could not unwind stack");
}

pub unsafe fn cleanup(ptr: *mut u8) -> Box<Any + Send + 'static> {
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

#[lang = "eh_personality_catch"]
#[cfg(not(test))]
unsafe extern fn rust_eh_personality_catch(
    exceptionRecord: *mut EXCEPTION_RECORD,
    establisherFrame: LPVOID,
    contextRecord: *mut CONTEXT,
    dispatcherContext: *mut DISPATCHER_CONTEXT
) -> EXCEPTION_DISPOSITION
{
    rust_eh_personality(exceptionRecord, establisherFrame,
                        contextRecord, dispatcherContext)
}

#[lang = "eh_personality"]
#[cfg(not(test))]
unsafe extern fn rust_eh_personality(
    exceptionRecord: *mut EXCEPTION_RECORD,
    establisherFrame: LPVOID,
    contextRecord: *mut CONTEXT,
    dispatcherContext: *mut DISPATCHER_CONTEXT
) -> EXCEPTION_DISPOSITION
{
    let er = &*exceptionRecord;
    let dc = &*dispatcherContext;

    if er.ExceptionFlags & EXCEPTION_UNWIND == 0 { // we are in the dispatch phase
        if er.ExceptionCode == RUST_PANIC {
            if let Some(lpad) = find_landing_pad(dc) {
                RtlUnwindEx(establisherFrame,
                            lpad as LPVOID,
                            exceptionRecord,
                            er.ExceptionInformation[0] as LPVOID, // pointer to PanicData
                            contextRecord,
                            dc.HistoryTable);
                rtabort!("could not unwind");
            }
        }
    }
    ExceptionContinueSearch
}

#[cfg(not(test))]
#[lang = "eh_unwind_resume"]
#[unwind]
unsafe extern fn rust_eh_unwind_resume(panic_ctx: LPVOID) -> ! {
    let params = [panic_ctx as ULONG_PTR];
    RaiseException(RUST_PANIC,
                   EXCEPTION_NONCONTINUABLE,
                   params.len() as DWORD,
                   &params as *const ULONG_PTR);
    rtabort!("could not resume unwind");
}

unsafe fn find_landing_pad(dc: &DISPATCHER_CONTEXT) -> Option<usize> {
    let eh_ctx = eh::EHContext {
        ip: dc.ControlPc as usize,
        func_start: dc.ImageBase as usize + (*dc.FunctionEntry).BeginAddress as usize,
        text_start: dc.ImageBase as usize,
        data_start: 0
    };
    eh::find_landing_pad(dc.HandlerData, &eh_ctx)
}
