//! A shim for libunwind implemented in terms of the native wasm `throw` instruction.

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

pub const unwinder_private_data_size: usize = 2;

#[repr(C)]
pub struct _Unwind_Exception {
    pub exception_class: _Unwind_Exception_Class,
    pub exception_cleanup: _Unwind_Exception_Cleanup_Fn,
    pub private: [_Unwind_Word; unwinder_private_data_size],
}

pub type _Unwind_Exception_Cleanup_Fn =
    Option<extern "C" fn(unwind_code: _Unwind_Reason_Code, exception: *mut _Unwind_Exception)>;

pub unsafe fn _Unwind_DeleteException(exception: *mut _Unwind_Exception) {
    if let Some(exception_cleanup) = unsafe { (*exception).exception_cleanup } {
        exception_cleanup(_URC_FOREIGN_EXCEPTION_CAUGHT, exception);
    }
}

pub unsafe fn _Unwind_RaiseException(exception: *mut _Unwind_Exception) -> _Unwind_Reason_Code {
    // The wasm `throw` instruction takes a "tag", which differentiates certain
    // types of exceptions from others. LLVM currently just identifies these
    // via integers, with 0 corresponding to C++ exceptions and 1 to C setjmp()/longjmp().
    // Ideally, we'd be able to choose something unique for Rust, but for now,
    // we pretend to be C++ and implement the Itanium exception-handling ABI.
    cfg_if::cfg_if! {
        // panic=abort is default for wasm targets. Because an unknown instruction is a load-time
        // error on wasm, instead of a runtime error like on traditional architectures, we never
        // want to codegen a `throw` instruction, as that would break users using runtimes that
        // don't yet support exceptions. The only time this first branch would be selected is if
        // the user explicitly opts in to wasm exceptions, via -Zbuild-std with -Cpanic=unwind.
        if #[cfg(panic = "unwind")] {
            // corresponds with llvm::WebAssembly::Tag::CPP_EXCEPTION
            //     in llvm-project/llvm/include/llvm/CodeGen/WasmEHFuncInfo.h
            const CPP_EXCEPTION_TAG: i32 = 0;
            core::arch::wasm::throw::<CPP_EXCEPTION_TAG>(exception.cast())
        } else {
            let _ = exception;
            core::arch::wasm::unreachable()
        }
    }
}
