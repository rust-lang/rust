#![allow(nonstandard_style)]

use core::ffi::{c_int, c_void};

pub type _Unwind_Action = c_int;

pub const _UA_SEARCH_PHASE: c_int = 1;
pub const _UA_CLEANUP_PHASE: c_int = 2;
pub const _UA_HANDLER_FRAME: c_int = 4;
pub const _UA_FORCE_UNWIND: c_int = 8;
pub const _UA_END_OF_STACK: c_int = 16;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
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
    _URC_FAILURE = 9, // used only by ARM EHABI
}
pub use _Unwind_Reason_Code::*;
pub use unwinding::abi::{UnwindContext, UnwindException};
pub enum _Unwind_Context {}

pub use unwinding::custom_eh_frame_finder::{
    EhFrameFinder, FrameInfo, FrameInfoKind, set_custom_eh_frame_finder,
};

pub type _Unwind_Exception_Class = u64;
pub type _Unwind_Word = *const u8;
pub type _Unwind_Ptr = *const u8;

pub const unwinder_private_data_size: usize = size_of::<UnwindException>()
    - size_of::<_Unwind_Exception_Class>()
    - size_of::<_Unwind_Exception_Cleanup_Fn>();

pub type _Unwind_Exception_Cleanup_Fn =
    Option<extern "C" fn(unwind_code: _Unwind_Reason_Code, exception: *mut _Unwind_Exception)>;

#[repr(C)]
pub struct _Unwind_Exception {
    pub exception_class: _Unwind_Exception_Class,
    pub exception_cleanup: _Unwind_Exception_Cleanup_Fn,
    pub private: [_Unwind_Word; unwinder_private_data_size],
}

pub unsafe fn _Unwind_GetDataRelBase(ctx: *mut _Unwind_Context) -> _Unwind_Ptr {
    let ctx = unsafe { &mut *(ctx as *mut UnwindContext<'_>) };
    unwinding::abi::_Unwind_GetDataRelBase(ctx) as _Unwind_Ptr
}

pub unsafe fn _Unwind_GetTextRelBase(ctx: *mut _Unwind_Context) -> _Unwind_Ptr {
    let ctx = unsafe { &mut *(ctx as *mut UnwindContext<'_>) };
    unwinding::abi::_Unwind_GetTextRelBase(ctx) as _Unwind_Ptr
}

pub unsafe fn _Unwind_GetRegionStart(ctx: *mut _Unwind_Context) -> _Unwind_Ptr {
    let ctx = unsafe { &mut *(ctx as *mut UnwindContext<'_>) };
    unwinding::abi::_Unwind_GetRegionStart(ctx) as _Unwind_Ptr
}

pub unsafe fn _Unwind_SetGR(ctx: *mut _Unwind_Context, reg_index: c_int, value: _Unwind_Word) {
    let ctx = unsafe { &mut *(ctx as *mut UnwindContext<'_>) };
    unwinding::abi::_Unwind_SetGR(ctx, reg_index, value as usize)
}

pub unsafe fn _Unwind_SetIP(ctx: *mut _Unwind_Context, value: _Unwind_Word) {
    let ctx = unsafe { &mut *(ctx as *mut UnwindContext<'_>) };
    unwinding::abi::_Unwind_SetIP(ctx, value as usize)
}

pub unsafe fn _Unwind_GetIPInfo(
    ctx: *mut _Unwind_Context,
    ip_before_insn: *mut c_int,
) -> _Unwind_Word {
    let ctx = unsafe { &mut *(ctx as *mut UnwindContext<'_>) };
    let ip_before_insn = unsafe { &mut *(ip_before_insn as *mut c_int) };
    unsafe { &*(unwinding::abi::_Unwind_GetIPInfo(ctx, ip_before_insn) as _Unwind_Word) }
}

pub unsafe fn _Unwind_GetLanguageSpecificData(ctx: *mut _Unwind_Context) -> *mut c_void {
    let ctx = unsafe { &mut *(ctx as *mut UnwindContext<'_>) };
    unwinding::abi::_Unwind_GetLanguageSpecificData(ctx)
}

pub unsafe fn _Unwind_RaiseException(exception: *mut _Unwind_Exception) -> _Unwind_Reason_Code {
    let exception = unsafe { &mut *(exception as *mut UnwindException) };
    unsafe { core::mem::transmute(unwinding::abi::_Unwind_RaiseException(exception)) }
}

pub unsafe fn _Unwind_DeleteException(exception: *mut _Unwind_Exception) {
    let exception = unsafe { &mut *(exception as *mut UnwindException) };
    unsafe { unwinding::abi::_Unwind_DeleteException(exception) }
}
