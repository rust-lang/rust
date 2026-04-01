//! On Emscripten Rust panics are wrapped in C++ exceptions, so we just forward
//! to `__gxx_personality_v0` which is provided by Emscripten.

use unwind as uw;

use crate::ffi::c_int;

// This is required by the compiler to exist (e.g., it's a lang item), but it's
// never actually called by the compiler.  Emscripten EH doesn't use a
// personality function at all, it instead uses __cxa_find_matching_catch.
// Wasm error handling would use __gxx_personality_wasm0.
#[lang = "eh_personality"]
unsafe extern "C" fn rust_eh_personality(
    _version: c_int,
    _actions: uw::_Unwind_Action,
    _exception_class: uw::_Unwind_Exception_Class,
    _exception_object: *mut uw::_Unwind_Exception,
    _context: *mut uw::_Unwind_Context,
) -> uw::_Unwind_Reason_Code {
    core::intrinsics::abort()
}
