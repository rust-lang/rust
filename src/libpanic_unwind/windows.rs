#![allow(nonstandard_style)]
#![allow(dead_code)]
#![cfg(windows)]

use libc::{c_long, c_ulong, c_void};

pub type DWORD = c_ulong;
pub type LONG = c_long;
pub type ULONG_PTR = usize;
pub type LPVOID = *mut c_void;

pub const EXCEPTION_MAXIMUM_PARAMETERS: usize = 15;
pub const EXCEPTION_NONCONTINUABLE: DWORD = 0x1;   // Noncontinuable exception
pub const EXCEPTION_UNWINDING: DWORD = 0x2;        // Unwind is in progress
pub const EXCEPTION_EXIT_UNWIND: DWORD = 0x4;      // Exit unwind is in progress
pub const EXCEPTION_TARGET_UNWIND: DWORD = 0x20;   // Target unwind in progress
pub const EXCEPTION_COLLIDED_UNWIND: DWORD = 0x40; // Collided exception handler call
pub const EXCEPTION_UNWIND: DWORD = EXCEPTION_UNWINDING | EXCEPTION_EXIT_UNWIND |
                                    EXCEPTION_TARGET_UNWIND |
                                    EXCEPTION_COLLIDED_UNWIND;

#[repr(C)]
pub struct EXCEPTION_RECORD {
    pub ExceptionCode: DWORD,
    pub ExceptionFlags: DWORD,
    pub ExceptionRecord: *mut EXCEPTION_RECORD,
    pub ExceptionAddress: LPVOID,
    pub NumberParameters: DWORD,
    pub ExceptionInformation: [LPVOID; EXCEPTION_MAXIMUM_PARAMETERS],
}

#[repr(C)]
pub struct EXCEPTION_POINTERS {
    pub ExceptionRecord: *mut EXCEPTION_RECORD,
    pub ContextRecord: *mut CONTEXT,
}

pub enum UNWIND_HISTORY_TABLE {}

#[repr(C)]
pub struct RUNTIME_FUNCTION {
    pub BeginAddress: DWORD,
    pub EndAddress: DWORD,
    pub UnwindData: DWORD,
}

pub enum CONTEXT {}

#[repr(C)]
pub struct DISPATCHER_CONTEXT {
    pub ControlPc: LPVOID,
    pub ImageBase: LPVOID,
    pub FunctionEntry: *const RUNTIME_FUNCTION,
    pub EstablisherFrame: LPVOID,
    pub TargetIp: LPVOID,
    pub ContextRecord: *const CONTEXT,
    pub LanguageHandler: LPVOID,
    pub HandlerData: *const u8,
    pub HistoryTable: *const UNWIND_HISTORY_TABLE,
}

#[repr(C)]
pub enum EXCEPTION_DISPOSITION {
    ExceptionContinueExecution,
    ExceptionContinueSearch,
    ExceptionNestedException,
    ExceptionCollidedUnwind,
}
pub use self::EXCEPTION_DISPOSITION::*;

extern "system" {
    #[unwind(allowed)]
    pub fn RaiseException(dwExceptionCode: DWORD,
                          dwExceptionFlags: DWORD,
                          nNumberOfArguments: DWORD,
                          lpArguments: *const ULONG_PTR);
    #[unwind(allowed)]
    pub fn RtlUnwindEx(TargetFrame: LPVOID,
                       TargetIp: LPVOID,
                       ExceptionRecord: *const EXCEPTION_RECORD,
                       ReturnValue: LPVOID,
                       OriginalContext: *const CONTEXT,
                       HistoryTable: *const UNWIND_HISTORY_TABLE);
    #[unwind(allowed)]
    pub fn _CxxThrowException(pExceptionObject: *mut c_void, pThrowInfo: *mut u8);
}
