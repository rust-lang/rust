#![allow(nonstandard_style)]

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

pub const unwinder_private_data_size: usize = cfg_select! {
    target_arch = "x86" => 5,
    all(target_arch = "x86_64", not(any(target_os = "windows", target_os = "cygwin"))) => 2,
    all(target_arch = "x86_64", any(target_os = "windows", target_os = "cygwin")) => 6,
    all(target_arch = "arm", not(target_vendor = "apple")) => 20,
    all(target_arch = "arm", target_vendor = "apple") => 5,
    all(target_arch = "aarch64", target_pointer_width = "64", not(target_os = "windows")) => 2,
    all(target_arch = "aarch64", target_pointer_width = "64", target_os = "windows") => 6,
    all(target_arch = "aarch64", target_pointer_width = "32") => 5,
    target_arch = "m68k" => 2,
    any(target_arch = "mips", target_arch = "mips32r6") => 2,
    target_arch = "csky" => 2,
    any(target_arch = "mips64", target_arch = "mips64r6") => 2,
    any(target_arch = "powerpc", target_arch = "powerpc64") => 2,
    target_arch = "s390x" => 2,
    any(target_arch = "sparc", target_arch = "sparc64") => 2,
    any(target_arch = "riscv64", target_arch = "riscv32") => 2,
    all(target_family = "wasm", target_os = "emscripten") => 20,
    target_family = "wasm" => 2,
    target_arch = "hexagon" => 5,
    any(target_arch = "loongarch32", target_arch = "loongarch64") => 2,
};

#[repr(C)]
pub struct _Unwind_Exception {
    pub exception_class: _Unwind_Exception_Class,
    pub exception_cleanup: _Unwind_Exception_Cleanup_Fn,
    pub private: [_Unwind_Word; unwinder_private_data_size],
}

// Check the size of _Unwind_Exception against the source of truth when using the unwinding crate.
#[cfg(target_os = "xous")]
const _: () = {
    assert!(size_of::<unwinding::abi::UnwindException>() == size_of::<_Unwind_Exception>());
};

pub type _Unwind_Exception_Cleanup_Fn =
    Option<extern "C" fn(unwind_code: _Unwind_Reason_Code, exception: *mut _Unwind_Exception)>;
