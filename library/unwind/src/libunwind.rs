#![allow(nonstandard_style)]

use core::ffi::{c_int, c_void};

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

pub type _Unwind_Exception_Class = u64;
pub type _Unwind_Word = *const u8;
pub type _Unwind_Ptr = *const u8;
pub type _Unwind_Trace_Fn =
    extern "C" fn(ctx: *mut _Unwind_Context, arg: *mut c_void) -> _Unwind_Reason_Code;

#[cfg(target_arch = "x86")]
pub const unwinder_private_data_size: usize = 5;

#[cfg(all(target_arch = "x86_64", not(any(target_os = "windows", target_os = "cygwin"))))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(all(target_arch = "x86_64", any(target_os = "windows", target_os = "cygwin")))]
pub const unwinder_private_data_size: usize = 6;

#[cfg(all(target_arch = "arm", not(target_vendor = "apple")))]
pub const unwinder_private_data_size: usize = 20;

#[cfg(all(target_arch = "arm", target_vendor = "apple"))]
pub const unwinder_private_data_size: usize = 5;

#[cfg(all(target_arch = "aarch64", target_pointer_width = "64", not(target_os = "windows")))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(all(target_arch = "aarch64", target_pointer_width = "64", target_os = "windows"))]
pub const unwinder_private_data_size: usize = 6;

#[cfg(all(target_arch = "aarch64", target_pointer_width = "32"))]
pub const unwinder_private_data_size: usize = 5;

#[cfg(target_arch = "m68k")]
pub const unwinder_private_data_size: usize = 2;

#[cfg(any(target_arch = "mips", target_arch = "mips32r6"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(target_arch = "csky")]
pub const unwinder_private_data_size: usize = 2;

#[cfg(any(target_arch = "mips64", target_arch = "mips64r6"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(target_arch = "s390x")]
pub const unwinder_private_data_size: usize = 2;

#[cfg(any(target_arch = "sparc", target_arch = "sparc64"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(all(target_arch = "wasm32", target_os = "emscripten"))]
pub const unwinder_private_data_size: usize = 20;

#[cfg(all(target_arch = "wasm32", target_os = "linux"))]
pub const unwinder_private_data_size: usize = 2;

#[cfg(all(target_arch = "hexagon", target_os = "linux"))]
pub const unwinder_private_data_size: usize = 35;

#[cfg(target_arch = "loongarch64")]
pub const unwinder_private_data_size: usize = 2;

#[repr(C)]
pub struct _Unwind_Exception {
    pub exception_class: _Unwind_Exception_Class,
    pub exception_cleanup: _Unwind_Exception_Cleanup_Fn,
    pub private: [_Unwind_Word; unwinder_private_data_size],
}

pub enum _Unwind_Context {}

pub type _Unwind_Exception_Cleanup_Fn =
    Option<extern "C" fn(unwind_code: _Unwind_Reason_Code, exception: *mut _Unwind_Exception)>;

// FIXME: The `#[link]` attributes on `extern "C"` block marks those symbols declared in
// the block are reexported in dylib build of std. This is needed when build rustc with
// feature `llvm-libunwind`, as no other cdylib will provided those _Unwind_* symbols.
// However the `link` attribute is duplicated multiple times and does not just export symbol,
// a better way to manually export symbol would be another attribute like `#[export]`.
// See the logic in function rustc_codegen_ssa::src::back::exported_symbols, module
// rustc_codegen_ssa::src::back::symbol_export, rustc_middle::middle::exported_symbols
// and RFC 2841
#[cfg_attr(
    all(
        feature = "llvm-libunwind",
        any(target_os = "fuchsia", target_os = "linux", target_os = "xous")
    ),
    link(name = "unwind", kind = "static", modifiers = "-bundle")
)]
unsafe extern "C-unwind" {
    pub fn _Unwind_Resume(exception: *mut _Unwind_Exception) -> !;
}
unsafe extern "C" {
    pub fn _Unwind_DeleteException(exception: *mut _Unwind_Exception);
    pub fn _Unwind_GetLanguageSpecificData(ctx: *mut _Unwind_Context) -> *mut c_void;
    pub fn _Unwind_GetRegionStart(ctx: *mut _Unwind_Context) -> _Unwind_Ptr;
    pub fn _Unwind_GetTextRelBase(ctx: *mut _Unwind_Context) -> _Unwind_Ptr;
    pub fn _Unwind_GetDataRelBase(ctx: *mut _Unwind_Context) -> _Unwind_Ptr;
}

cfg_if::cfg_if! {
if #[cfg(any(target_vendor = "apple", target_os = "netbsd", not(target_arch = "arm")))] {
    // Not ARM EHABI
    //
    // 32-bit ARM on iOS/tvOS/watchOS use either DWARF/Compact unwinding or
    // "setjmp-longjmp" / SjLj unwinding.
    pub type _Unwind_Action = c_int;

    pub const _UA_SEARCH_PHASE: c_int = 1;
    pub const _UA_CLEANUP_PHASE: c_int = 2;
    pub const _UA_HANDLER_FRAME: c_int = 4;
    pub const _UA_FORCE_UNWIND: c_int = 8;
    pub const _UA_END_OF_STACK: c_int = 16;

    #[cfg_attr(
        all(feature = "llvm-libunwind", any(target_os = "fuchsia", target_os = "linux", target_os = "xous")),
        link(name = "unwind", kind = "static", modifiers = "-bundle")
    )]
    unsafe extern "C" {
        pub fn _Unwind_GetGR(ctx: *mut _Unwind_Context, reg_index: c_int) -> _Unwind_Word;
        pub fn _Unwind_SetGR(ctx: *mut _Unwind_Context, reg_index: c_int, value: _Unwind_Word);
        pub fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> _Unwind_Word;
        pub fn _Unwind_SetIP(ctx: *mut _Unwind_Context, value: _Unwind_Word);
        pub fn _Unwind_GetIPInfo(ctx: *mut _Unwind_Context, ip_before_insn: *mut c_int)
                                 -> _Unwind_Word;
        pub fn _Unwind_FindEnclosingFunction(pc: *mut c_void) -> *mut c_void;
    }

} else {
    // ARM EHABI
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq)]
    pub enum _Unwind_State {
        _US_VIRTUAL_UNWIND_FRAME = 0,
        _US_UNWIND_FRAME_STARTING = 1,
        _US_UNWIND_FRAME_RESUME = 2,
        _US_ACTION_MASK = 3,
        _US_FORCE_UNWIND = 8,
        _US_END_OF_STACK = 16,
    }
    pub use _Unwind_State::*;

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
    use _Unwind_VRS_RegClass::*;
    #[repr(C)]
    enum _Unwind_VRS_DataRepresentation {
        _UVRSD_UINT32 = 0,
        _UVRSD_VFPX = 1,
        _UVRSD_FPAX = 2,
        _UVRSD_UINT64 = 3,
        _UVRSD_FLOAT = 4,
        _UVRSD_DOUBLE = 5,
    }
    use _Unwind_VRS_DataRepresentation::*;

    pub const UNWIND_POINTER_REG: c_int = 12;
    pub const UNWIND_SP_REG: c_int = 13;
    pub const UNWIND_IP_REG: c_int = 15;

    #[cfg_attr(
        all(feature = "llvm-libunwind", any(target_os = "fuchsia", target_os = "linux", target_os = "xous")),
        link(name = "unwind", kind = "static", modifiers = "-bundle")
    )]
    unsafe extern "C" {
        fn _Unwind_VRS_Get(ctx: *mut _Unwind_Context,
                           regclass: _Unwind_VRS_RegClass,
                           regno: _Unwind_Word,
                           repr: _Unwind_VRS_DataRepresentation,
                           data: *mut c_void)
                           -> _Unwind_VRS_Result;

        fn _Unwind_VRS_Set(ctx: *mut _Unwind_Context,
                           regclass: _Unwind_VRS_RegClass,
                           regno: _Unwind_Word,
                           repr: _Unwind_VRS_DataRepresentation,
                           data: *mut c_void)
                           -> _Unwind_VRS_Result;
    }

    // On Android or ARM/Linux, these are implemented as macros:

    pub unsafe fn _Unwind_GetGR(ctx: *mut _Unwind_Context, reg_index: c_int) -> _Unwind_Word {
        let mut val: _Unwind_Word = core::ptr::null();
        unsafe { _Unwind_VRS_Get(ctx, _UVRSC_CORE, reg_index as _Unwind_Word, _UVRSD_UINT32,
                        (&raw mut val) as *mut c_void); }
        val
    }

    pub unsafe fn _Unwind_SetGR(ctx: *mut _Unwind_Context, reg_index: c_int, value: _Unwind_Word) {
        let mut value = value;
        unsafe { _Unwind_VRS_Set(ctx, _UVRSC_CORE, reg_index as _Unwind_Word, _UVRSD_UINT32,
                        (&raw mut value) as *mut c_void); }
    }

    pub unsafe fn _Unwind_GetIP(ctx: *mut _Unwind_Context)
                                -> _Unwind_Word {
        let val = unsafe { _Unwind_GetGR(ctx, UNWIND_IP_REG) };
        val.map_addr(|v| v & !1)
    }

    pub unsafe fn _Unwind_SetIP(ctx: *mut _Unwind_Context,
                                value: _Unwind_Word) {
        // Propagate thumb bit to instruction pointer
        let thumb_state = unsafe { _Unwind_GetGR(ctx, UNWIND_IP_REG).addr() & 1 };
        let value = value.map_addr(|v| v | thumb_state);
        unsafe { _Unwind_SetGR(ctx, UNWIND_IP_REG, value); }
    }

    pub unsafe fn _Unwind_GetIPInfo(ctx: *mut _Unwind_Context,
                                    ip_before_insn: *mut c_int)
                                    -> _Unwind_Word {
        unsafe {
            *ip_before_insn = 0;
            _Unwind_GetIP(ctx)
        }
    }

    // This function also doesn't exist on Android or ARM/Linux, so make it a no-op
    pub unsafe fn _Unwind_FindEnclosingFunction(pc: *mut c_void) -> *mut c_void {
        pc
    }
}
} // cfg_if!

cfg_if::cfg_if! {
if #[cfg(all(target_vendor = "apple", not(target_os = "watchos"), target_arch = "arm"))] {
    // 32-bit ARM Apple (except for watchOS armv7k specifically) uses SjLj and
    // does not provide _Unwind_Backtrace()
    unsafe extern "C-unwind" {
        pub fn _Unwind_SjLj_RaiseException(e: *mut _Unwind_Exception) -> _Unwind_Reason_Code;
    }

    pub use _Unwind_SjLj_RaiseException as _Unwind_RaiseException;
} else {
    #[cfg_attr(
        all(feature = "llvm-libunwind", any(target_os = "fuchsia", target_os = "linux", target_os = "xous")),
        link(name = "unwind", kind = "static", modifiers = "-bundle")
    )]
    unsafe extern "C-unwind" {
        pub fn _Unwind_RaiseException(exception: *mut _Unwind_Exception) -> _Unwind_Reason_Code;
    }
    #[cfg_attr(
        all(feature = "llvm-libunwind", any(target_os = "fuchsia", target_os = "linux", target_os = "xous")),
        link(name = "unwind", kind = "static", modifiers = "-bundle")
    )]
    unsafe extern "C" {
        pub fn _Unwind_Backtrace(trace: _Unwind_Trace_Fn,
                                 trace_argument: *mut c_void)
                                 -> _Unwind_Reason_Code;
    }
}
} // cfg_if!

cfg_if::cfg_if! {
if #[cfg(any(
        all(windows, any(target_arch = "aarch64", target_arch = "x86_64"), target_env = "gnu"),
        target_os = "cygwin",
    ))] {
    // We declare these as opaque types. This is fine since you just need to
    // pass them to _GCC_specific_handler and forget about them.
    pub enum EXCEPTION_RECORD {}
    pub type LPVOID = *mut c_void;
    pub enum CONTEXT {}
    pub enum DISPATCHER_CONTEXT {}
    pub type EXCEPTION_DISPOSITION = c_int;
    type PersonalityFn = unsafe extern "C" fn(version: c_int,
                                              actions: _Unwind_Action,
                                              exception_class: _Unwind_Exception_Class,
                                              exception_object: *mut _Unwind_Exception,
                                              context: *mut _Unwind_Context)
                                              -> _Unwind_Reason_Code;

    unsafe extern "C" {
        pub fn _GCC_specific_handler(exceptionRecord: *mut EXCEPTION_RECORD,
                                establisherFrame: LPVOID,
                                contextRecord: *mut CONTEXT,
                                dispatcherContext: *mut DISPATCHER_CONTEXT,
                                personality: PersonalityFn)
                                -> EXCEPTION_DISPOSITION;
    }
}
} // cfg_if!
