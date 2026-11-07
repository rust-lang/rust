#![allow(nonstandard_style)]

// Use the unwinding crate as unwinder on Xous
#[cfg(target_os = "xous")]
pub use unwinding::custom_eh_frame_finder::{
    EhFrameFinder, FrameInfo, FrameInfoKind, set_custom_eh_frame_finder,
};

pub use crate::types::*;

// FIXME: The `#[link]` attributes on `extern "C"` block marks those symbols declared in
// the block are reexported in dylib build of std. This is needed when build rustc with
// feature `llvm-libunwind`, as no other cdylib will provided those _Unwind_* symbols.
// However the `link` attribute is duplicated multiple times and does not just export symbol,
// a better way to manually export symbol would be another attribute like `#[export]`.
// See the logic in function rustc_codegen_ssa::src::back::exported_symbols, module
// rustc_codegen_ssa::src::back::symbol_export, rustc_middle::middle::exported_symbols
// and RFC 2841
#[cfg_attr(
    all(feature = "llvm-libunwind", any(target_os = "fuchsia", target_os = "linux")),
    link(name = "unwind", kind = "static", modifiers = "-bundle")
)]
unsafe extern "C-unwind" {
    pub(crate) fn _Unwind_Resume(exception: *mut _Unwind_Exception) -> !;
}
unsafe extern "C" {
    pub fn _Unwind_DeleteException(exception: *mut _Unwind_Exception);
}

#[cfg_attr(
    all(feature = "llvm-libunwind", any(target_os = "fuchsia", target_os = "linux")),
    link(name = "unwind", kind = "static", modifiers = "-bundle")
)]
unsafe extern "C-unwind" {
    // 32-bit ARM Apple (except for watchOS armv7k specifically) uses SjLj
    #[cfg_attr(
        all(target_vendor = "apple", not(target_os = "watchos"), target_arch = "arm"),
        link_name = "_Unwind_SjLj_RaiseException"
    )]
    pub fn _Unwind_RaiseException(exception: *mut _Unwind_Exception) -> _Unwind_Reason_Code;
}
