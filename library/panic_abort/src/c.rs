#![allow(nonstandard_style)]

use libc::{c_int, c_ulong, c_void};

pub type DWORD = c_ulong;
pub type BOOL = c_int;

pub type LPVOID = *mut c_void;

pub const FALSE: BOOL = 0;

pub const PF_FASTFAIL_AVAILABLE: DWORD = 23;

pub const EXCEPTION_MAXIMUM_PARAMETERS: usize = 15;

#[repr(C)]
pub struct EXCEPTION_RECORD {
    pub ExceptionCode: DWORD,
    pub ExceptionFlags: DWORD,
    pub ExceptionRecord: *mut EXCEPTION_RECORD,
    pub ExceptionAddress: LPVOID,
    pub NumberParameters: DWORD,
    pub ExceptionInformation: [LPVOID; EXCEPTION_MAXIMUM_PARAMETERS],
}

pub enum CONTEXT {}

extern "system" {
    pub fn IsProcessorFeaturePresent(ProcessorFeature: DWORD) -> BOOL;
    pub fn RaiseFailFastException(
        pExceptionRecord: *const EXCEPTION_RECORD,
        pContextRecord: *const CONTEXT,
        dwFlags: DWORD,
    );
}
